import copy
import csv
from functools import reduce
from pathlib import Path
from typing import Any, Tuple

import catenets.models as cate_models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

import src.iterpretability.logger as log
from src.iterpretability.explain import Explainer
from src.iterpretability.datasets.data_loader import load
from src.iterpretability.synthetic_simulate import SyntheticSimulatorLinear
from src.iterpretability.utils import (
    attribution_accuracy,
    compute_cate_metrics,
    dataframe_line_plot,
)


class PredictiveSensitivity:
    """
    Sensitivity analysis for Confounding
    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        n_iter: int = 1000,
        seed: int = 42,
        explainer_limit: int = 100,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 1, 10, 100],
        binary_outcome: bool = False,
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.predictive_scales = predictive_scales
        self.binary_outcome = binary_outcome

    def run(
        self,
        dataset: str = "tcga_10",
        num_important_features: int = 2,
        explainer_list: list = ["feature_ablation", "feature_permutation", "integrated_gradients",
                                "shapley_value_sampling", "lime"],
    ) -> None:
        log.info(f"Using dataset {dataset} with num_important features = {num_important_features}.")

        X_raw_train, X_raw_test = load(dataset, train_ratio=0.8)
        sim = SyntheticSimulatorLinear(X_raw_train, num_important_features=num_important_features)

        explainability_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            X_train, W_train, Y_train, po0_train, po1_train, propensity_train = sim.simulate_dataset(X_raw_train,
                                                                                                     predictive_scale=predictive_scale,
                                                                                                     binary_outcome=self.binary_outcome)

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(X_raw_test,
                                                                                 predictive_scale=predictive_scale,
                                                                                 binary_outcome=self.binary_outcome)

            log.info("Fitting and explaining learners...")
            learners = {
                "TLearner": cate_models.torch.TLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "SLearner": cate_models.torch.SLearner(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "TARNet": cate_models.torch.TARNet(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_r=1,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "SNet": cate_models.torch.SNet(
                    X_train.shape[1],
                    binary_y=(len(np.unique(Y_train)) == 2),
                    n_layers_r=1,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=50,
                    n_units_r_small=50,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    penalty_orthogonal=0.01,
                    nonlin="relu",
                ),
            }

            learner_explainers = {}
            learner_explanations = {}

            for name in learners:
                learners[name].fit(X=X_train, y=Y_train, w=W_train)
                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X_train.shape[1])),
                    explainer_list=explainer_list,
                )
                learner_explanations[name] = learner_explainers[name].explain(X_test[:self.explainer_limit])

            all_important_features = reduce(np.union1d, (
                np.where((sim.prog_mask).astype(np.int32) != 0)[0],
                np.where((sim.pred0_mask).astype(np.int32) != 0)[0],
                np.where((sim.pred1_mask).astype(np.int32) != 0)[0]))

            pred_features = np.union1d(np.where((sim.pred0_mask).astype(np.int32) != 0)[0],
                                       np.where((sim.pred1_mask).astype(np.int32) != 0)[0])

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(learner_explanations[learner_name][explainer_name])
                    acc_scores_all_features = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features = attribution_accuracy(
                        pred_features, attribution_est
                    )
                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_predictive_features,
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features acc%",
                "Pred features acc%",
            ],
        )

        results_path = self.save_path / "results/"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_binary_{self.binary_outcome}-seed{self.seed}.csv")
