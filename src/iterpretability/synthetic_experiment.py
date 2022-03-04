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
from src.iterpretability.synthetic_simulate import SyntheticSimulatorLinear, SyntheticSimulatorLinearCorrelations, SyntheticSimulatorLinearPairwise
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
        explainer_limit: int = 500,
        save_path: Path = Path.cwd(),
        predictive_scales: list = [1e-3, 1e-2, 1e-1, 1, 2, 5, 10],
        num_interactions: int = 1,
        synthetic_simulator_type: str = 'linear',
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
        self.num_interactions = num_interactions
        self.synthetic_simulator_type = synthetic_simulator_type

    def run(
        self,
        dataset: str = "tcga_10",
        train_ratio: float = 0.8,
        num_important_features: int = 2,
        binary_outcome: bool = False,
        random_feature_selection: bool = False,
        explainer_list: list = ["feature_ablation", "feature_permutation", "integrated_gradients",
                                "shapley_value_sampling", "lime"],
    ) -> None:
        log.info(f"Using dataset {dataset} with num_important features = {num_important_features}.")

        X_raw_train, X_raw_test = load(dataset, train_ratio=train_ratio)

        if self.synthetic_simulator_type == 'linear':
            sim = SyntheticSimulatorLinear(X_raw_train, num_important_features=num_important_features,
                                           random_feature_selection=random_feature_selection, seed=self.seed)

        elif self.synthetic_simulator_type == 'linear_least_correlated':
            sim = SyntheticSimulatorLinearCorrelations(X_raw_train, num_important_features=num_important_features,
                                                       correlation_type='least_correlated', seed=self.seed)

        elif self.synthetic_simulator_type == 'linear_most_correlated':
            sim = SyntheticSimulatorLinearCorrelations(X_raw_train, num_important_features=num_important_features,
                                                       correlation_type='most_correlated', seed=self.seed)

        elif self.synthetic_simulator_type == 'linear_pairwise_interactions':
            sim = SyntheticSimulatorLinearPairwise(X_raw_train, num_important_features=num_important_features,
                                                   num_interactions = self.num_interactions, seed=self.seed)
        else:
            raise Exception('Unknown simulator type.')

        explainability_data = []

        for predictive_scale in self.predictive_scales:
            log.info(f"Now working with predictive_scale = {predictive_scale}...")
            X_train, W_train, Y_train, po0_train, po1_train, propensity_train = sim.simulate_dataset(X_raw_train,
                                                                                                     predictive_scale=predictive_scale,
                                                                                                     binary_outcome=binary_outcome)

            X_test, W_test, Y_test, po0_test, po1_test, _ = sim.simulate_dataset(X_raw_test,
                                                                                 predictive_scale=predictive_scale,
                                                                                 binary_outcome=binary_outcome)

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

            all_important_features = sim.get_all_important_features()
            pred_features = sim.get_predictive_features()
            prog_features = sim.get_prognostic_features()

            cate_test = sim.te(X_test)

            for explainer_name in explainer_list:
                for learner_name in learners:
                    attribution_est = np.abs(learner_explanations[learner_name][explainer_name])
                    acc_scores_all_features = attribution_accuracy(
                        all_important_features, attribution_est
                    )
                    acc_scores_predictive_features = attribution_accuracy(pred_features, attribution_est)
                    acc_scores_prog_features = attribution_accuracy(prog_features, attribution_est)
                    _, mu0_pred, mu1_pred = learners[learner_name].predict(X=X_test, return_po=True)

                    pehe_test, factual_rmse_test = compute_cate_metrics(cate_true=cate_test, y_true=Y_test,
                                                                        w_true=W_test,
                                                                        mu0_pred=mu0_pred, mu1_pred=mu1_pred)

                    explainability_data.append(
                        [
                            predictive_scale,
                            learner_name,
                            explainer_name,
                            acc_scores_all_features,
                            acc_scores_predictive_features,
                            acc_scores_prog_features,
                            pehe_test,
                            factual_rmse_test,
                            np.mean(cate_test),
                            np.var(cate_test),
                            pehe_test / np.sqrt(np.var(cate_test))
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Predictive Scale",
                "Learner",
                "Explainer",
                "All features ACC",
                "Pred features ACC",
                "Prog features ACC",
                "PEHE",
                "Factual RMSE",
                "CATE true mean",
                "CATE true var",
                "Normalized PEHE",
            ],

        )

        results_path = self.save_path / "results/"
        log.info(f"Saving results in {results_path}...")
        if not results_path.exists():
            results_path.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(
            results_path / f"predictive_scale_{dataset}_{num_important_features}_{self.synthetic_simulator_type}_random_{random_feature_selection}_binary_{binary_outcome}-seed{self.seed}.csv")
