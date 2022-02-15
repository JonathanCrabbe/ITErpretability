import copy
import csv
from pathlib import Path
from typing import Any, Tuple

import catenets.models as cate_models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catenets.datasets import load
from sklearn.metrics import mean_squared_error

import iterpretability.logger as log
from iterpretability.explain import Explainer
from iterpretability.simulate import Simulator
from iterpretability.utils import (
    DgpFunctionWrapper,
    PotentialOutcomeWrapper,
    abbrev_dict,
    attribution_fractions,
    compute_cate_metrics,
    dataframe_line_plot,
)


class Experiment:
    def __init__(
        self,
        learners: dict,
    ) -> None:
        assert isinstance(learners, dict)
        self.learners = learners

    def run(
        self,
        X_raw: pd.DataFrame,
        T_raw: pd.DataFrame,
        Y_raw: pd.DataFrame,
        explainer_limit: int = 100,
        **dgp_params: Any,
    ) -> Tuple:
        """
        Run the experiment

        Args:
            X_raw: baseline dataframe/numpy array of covariates
            T_raw: baseline dataframe/numpy array of treatments
            Y_raw: baseline dataframe/numpy array of outcomes
            explainer_limit: number of rows to explain
            dgp_params: Parameters to be forwarded to the DGP

        Returns:
            prog_explanations: Feature importance based on the prognostic model in the DGP
            pred_explanations: Feature importance based on the predictive model in the DGP
            explanations: Feature importance for the treatment effects based on the trained models
            po_explanations: Feature importance for the potential outcomes based on the trained models
        """
        sim = Simulator(X_raw, T_raw, Y_raw, **dgp_params)
        X, W, Y, po0, po1, prop, dgp_model, _ = sim.simulate_dataset(X_raw)

        log.info("Generate the feature importance for the prognostic model")
        prog_explainer = Explainer(
            dgp_model.prog, feature_names=list(range(X.shape[1]))
        )

        prog_explanations = prog_explainer.explain(X[:explainer_limit])

        log.info(
            "Generate the feature importance for the predictive model(po1- po0 importance)"
        )
        pred_explainer = Explainer(dgp_model, feature_names=list(range(X.shape[1])))

        pred_explanations = pred_explainer.explain(X[:explainer_limit])

        explainers = {}
        local_learners = {}

        for name in self.learners:
            local_learners[name] = copy.deepcopy(self.learners[name])
            local_learners[name].fit(X=X, y=Y, w=W)
            explainers[name] = Explainer(
                local_learners[name], feature_names=list(range(X.shape[1]))
            )

        log.info("Train explainers for potential outcomes")
        po_explainers = {}

        for name in local_learners:
            po0_wrapper = PotentialOutcomeWrapper(local_learners[name], po=0)
            po1_wrapper = PotentialOutcomeWrapper(local_learners[name], po=1)

            po_explainers[name] = (
                Explainer(po0_wrapper, feature_names=list(range(X.shape[1]))),
                Explainer(po1_wrapper, feature_names=list(range(X.shape[1]))),
            )

        log.info("Evaluate the feature importance")

        explanations = {}
        po_explanations = {}

        for src in explainers:  # For each learner explainer: SNet, TLearner etc.
            explanations[src] = explainers[src].explain(X[:explainer_limit])

        for src in po_explainers:  # For each learner explainer: SNet, TLearner etc.
            explanations[src] = explainers[src].explain(X[:explainer_limit])
            po_explanations[src] = (
                po_explainers[src][0].explain(X[:explainer_limit]),
                po_explainers[src][1].explain(X[:explainer_limit]),
            )

        return prog_explanations, pred_explanations, explanations, po_explanations


class ExplainabilityPlots:
    """

    Predictive-Prognostic Plots for the DGP without Learner

    """

    def __init__(
        self,
        n_units_hidden: int = 50,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        scale_factor: float = 10,
        n_iter: int = 3500,
        seed: int = 42,
        explainer_limit: int = 100,
        prognostic_mask: float = 0.1,
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.prognostic_mask = prognostic_mask

    def run(
        self, dataset: str = "twins", explainer_list: list = ["integrated_gradients"]
    ) -> None:
        X_raw, T_raw, Y_raw, _, _, _ = load(dataset)

        sim = Simulator(
            X_raw,
            T_raw,
            Y_raw,
            n_units_hidden=self.n_units_hidden,
            n_layers=self.n_layers,
            penalty_orthogonal=self.penalty_orthogonal,
            batch_size=self.batch_size,
            n_iter=self.n_iter,
            seed=self.seed,
        )
        X, W, Y, po0, po1, prop, dgp_model, _ = sim.simulate_dataset(
            X_raw, scale_factor=self.scale_factor, prognostic_mask=self.prognostic_mask
        )

        limit = self.explainer_limit
        dgp_te = DgpFunctionWrapper(sim, "te")
        dgp_prog = DgpFunctionWrapper(sim, "prog")
        dgp_po0 = DgpFunctionWrapper(sim, "po0")
        dgp_po1 = DgpFunctionWrapper(sim, "po1")

        log.info("Generate the feature importance for the prognostic part of the DGP")
        prog_explainer = Explainer(
            dgp_prog,
            feature_names=list(range(X.shape[1])),
            explainer_list=explainer_list,
        )
        prog_explanations = prog_explainer.explain(X[:limit])

        log.info("Generate the feature importance for the predictive part of the DGP")
        pred_explainer = Explainer(
            dgp_te, feature_names=list(range(X.shape[1])), explainer_list=explainer_list
        )
        pred_explanations = pred_explainer.explain(X[:limit])

        log.info(
            "Generate the feature importance for the potential outcomes from the DGP"
        )
        po0_explainer = Explainer(
            dgp_po0,
            feature_names=list(range(X.shape[1])),
            explainer_list=explainer_list,
        )
        po1_explainer = Explainer(
            dgp_po1,
            feature_names=list(range(X.shape[1])),
            explainer_list=explainer_list,
        )
        po0_explanations = po0_explainer.explain(X[:limit])
        po1_explanations = po1_explainer.explain(X[:limit])

        color_palette = sns.blend_palette(
            ["black", "red", "yellow", "green", "blue", "magenta", "pink"],
            n_colors=X.shape[1],
        )
        sns.set(font_scale=1.2)

        for explainer in explainer_list:
            pred_truth = pred_explanations[explainer]
            prog_truth = prog_explanations[explainer]
            po0_truth = po0_explanations[explainer]
            po1_truth = po1_explanations[explainer]

            linearity_violation = mean_squared_error(pred_truth, po1_truth - po0_truth)

            fig, axs = plt.subplots(1, 3, figsize=(25, 5))

            # Make the Predictive and Prognostic importance plots
            sns.boxplot(data=prog_truth, ax=axs[0], palette=color_palette)
            axs[0].set_title(f"Prognostic Importance - {abbrev_dict[explainer]}")
            axs[0].set_xlabel("Covariate Number")
            axs[0].set_ylabel("Prognostic Importance")
            sns.boxplot(data=pred_truth, ax=axs[1], palette=color_palette)
            axs[1].set_title(f"Predictive Importance - {abbrev_dict[explainer]}")
            axs[1].set_xlabel("Covariate Number")
            axs[1].set_ylabel("Predictive Importance")

            # Hide some of the ticks
            for ind, label in enumerate(axs[0].get_xticklabels()):
                if ind % 5 == 0:  # every 10th label is kept
                    label.set_visible(True)
                else:
                    label.set_visible(False)
            for ind, label in enumerate(axs[1].get_xticklabels()):
                if ind % 5 == 0:  # every 10th label is kept
                    label.set_visible(True)
                else:
                    label.set_visible(False)

            # Put the PO importance scores in a dataframe
            po_list = []
            for covariate in range(pred_truth.shape[1]):
                for example in range(pred_truth.shape[0]):
                    po0_covariate_score = po0_truth[example, covariate]
                    po1_covariate_score = po1_truth[example, covariate]
                    po_list.append(
                        [po0_covariate_score, po1_covariate_score, covariate]
                    )
            po_df = pd.DataFrame(
                po_list, columns=["PO0_Score", "PO1_Score", "Covariate"]
            )

            # Make the PO importance plots
            sns.scatterplot(
                data=po_df,
                x="PO0_Score",
                y="PO1_Score",
                hue="Covariate",
                ax=axs[2],
                palette=color_palette,
            )
            axs[2].get_legend().remove()
            x_vals = np.array(axs[2].get_xlim())
            axs[2].plot(x_vals, x_vals, "--")
            axs[2].set_title(
                f"Potential Outcomes Importance - {abbrev_dict[explainer]}"
            )
            axs[2].set_xlabel("PO_0 Importance")
            axs[2].set_ylabel("PO_1 Importance")

            log.info(f"Explainer = {explainer}")
            log.info(f"Linearity Violation = {linearity_violation:.2g}")
            plt.show()


class ExplainabilityMetrics:
    """

    Testing metrics for covariate importance

    """

    def __init__(
        self,
        n_units_hidden: int = 200,
        n_layers: int = 2,
        penalty_orthogonal: float = 0.01,
        batch_size: int = 1024,
        scale_factor: float = 100,
        n_iter: int = 3500,
        seed: int = 44,
        explainer_limit: int = 100,
        prognostic_mask: float = 0.01,
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.prognostic_mask = prognostic_mask

    def run(
        self, dataset: str = "twins", explainer_list: list = ["integrated_gradients"]
    ) -> None:

        X_raw, T_raw, Y_raw, _, _, _ = load(dataset)

        print("Fitting DGP...")
        sim = Simulator(
            X_raw,
            T_raw,
            Y_raw,
            n_units_hidden=self.n_units_hidden,
            n_layers=self.n_layers,
            penalty_orthogonal=self.penalty_orthogonal,
            batch_size=self.batch_size,
            n_iter=self.n_iter,
            seed=self.seed,
        )
        X, W, Y, po0, po1, prop, dgp_model = sim.simulate_dataset(
            X_raw, scale_factor=self.scale_factor, prognostic_mask=self.prognostic_mask
        )

        dgp_te = DgpFunctionWrapper(sim, "te")
        dgp_prog = DgpFunctionWrapper(sim, "prog")

        print("Explaining DGP...")
        limit = self.explainer_limit
        # Generate the feature importance for the prognostic model
        prog_explainer = Explainer(
            dgp_prog,
            feature_names=list(range(X.shape[1])),
            explainer_list=explainer_list,
        )
        prog_explanations = prog_explainer.explain(X[:limit])

        # Generate the feature importance for the predictive model(po1- po0 importance)
        pred_explainer = Explainer(
            dgp_te, feature_names=list(range(X.shape[1])), explainer_list=explainer_list
        )
        pred_explanations = pred_explainer.explain(X[:limit])

        print("Fitting Learners...")
        learners = {
            "TLearner": cate_models.torch.TLearner(
                X.shape[1],
                binary_y=(len(np.unique(Y)) == 2),
                n_layers_out=3,
                n_units_out=200,
                batch_size=1024,
                n_iter=1000,
                batch_norm=False,
            ),
            "SLearner": cate_models.torch.SLearner(
                X.shape[1],
                binary_y=(len(np.unique(Y)) == 2),
                n_layers_out=3,
                n_units_out=200,
                n_iter=1000,
                batch_size=1024,
                batch_norm=False,
            ),
            "TARNet": cate_models.torch.TARNet(
                X.shape[1],
                binary_y=(len(np.unique(Y)) == 2),
                n_layers_r=2,
                n_layers_out=1,
                n_units_out=200,
                batch_size=1024,
                n_iter=1000,
                batch_norm=False,
            ),
            "SNet": cate_models.torch.SNet(
                X.shape[1],
                binary_y=(len(np.unique(Y)) == 2),
                n_layers_r=2,
                n_layers_out=1,
                n_units_out=200,
                batch_size=1024,
                n_iter=1000,
                batch_norm=False,
                penalty_orthogonal=0.01,
            ),
        }

        learner_explainers = {}
        learner_explanations = {}

        for name in learners:
            learners[name].fit(X=X, y=Y, w=W)
            learner_explainers[name] = Explainer(
                learners[name],
                feature_names=list(range(X.shape[1])),
                explainer_list=explainer_list,
            )
            learner_explanations[name] = learner_explainers[name].explain(X[:limit])

        print("Explaining Learners...")

        xai_metrics_list = []
        for explainer_name in explainer_list:
            attribution_pred = pred_explanations[explainer_name]
            attribution_prog = prog_explanations[explainer_name]
            for learner_name in learners:
                attribution_est = learner_explanations[learner_name][explainer_name]
                pred_scores = attribution_fractions(attribution_pred, attribution_est)
                prog_scores = attribution_fractions(attribution_prog, attribution_est)
                xai_metrics_list.append(
                    [
                        learner_name,
                        abbrev_dict[explainer_name],
                        np.mean(pred_scores),
                        np.mean(prog_scores),
                    ]
                )

        xai_df = pd.DataFrame(
            data=xai_metrics_list,
            columns=["learner", "explainer", "pred_fraction", "prog_fraction"],
        )
        print(25 * "_" + "Fraction of predictive covariates identified" + 25 * "_")
        print(
            pd.crosstab(
                xai_df.learner, xai_df.explainer, xai_df.pred_fraction, aggfunc="mean"
            )
        )
        print(25 * "_" + "Fraction of prognostic covariates identified" + 25 * "_")
        print(
            pd.crosstab(
                xai_df.learner, xai_df.explainer, xai_df.prog_fraction, aggfunc="mean"
            )
        )


class PrognosticSensitivity:
    """
    Sensitivity analysis for the prognostic mask
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
        prognostic_masks: list = [1e-3, 1e-2, 1e-1, 1],
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.prognostic_masks = prognostic_masks

    def run(
        self,
        dataset: str = "twins",
        explainer_list: list = ["integrated_gradients"],
    ) -> None:

        X_raw, T_raw, Y_raw, _, _, _ = load(dataset, train_ratio=1.0)

        log.info("Fitting DGP...")
        sim = Simulator(
            X_raw,
            T_raw,
            Y_raw,
            n_units_hidden=self.n_units_hidden,
            n_layers=self.n_layers,
            penalty_orthogonal=self.penalty_orthogonal,
            batch_size=self.batch_size,
            n_iter=self.n_iter,
            seed=self.seed,
        )

        explainability_data = []

        for prognostic_mask in self.prognostic_masks:
            log.info(f"Now working with prognostic mask = {prognostic_mask}...")
            X, W, Y, po0, po1, prop, dgp_model, _ = sim.simulate_dataset(
                X_raw,
                prognostic_mask=prognostic_mask,
                scale_factor=10,
                noise=True,
                err_std=0.1,
            )

            dgp_te = DgpFunctionWrapper(sim, "te")
            dgp_prog = DgpFunctionWrapper(sim, "prog")

            log.info("Explaining DGP...")
            limit = self.explainer_limit
            # Generate the feature importance for the prognostic model
            prog_explainer = Explainer(
                dgp_prog,
                feature_names=list(range(X.shape[1])),
                explainer_list=explainer_list,
            )
            prog_explanations = prog_explainer.explain(X[:limit])
            # Generate the feature importance for the predictive model(po1- po0 importance)
            pred_explainer = Explainer(
                dgp_te,
                feature_names=list(range(X.shape[1])),
                explainer_list=explainer_list,
            )
            pred_explanations = pred_explainer.explain(X[:limit])

            log.info("Fitting and explaining learners...")
            learners = {
                "TLearner": cate_models.torch.TLearner(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "SLearner": cate_models.torch.SLearner(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_out=2,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "TARNet": cate_models.torch.TARNet(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
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
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
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
                learners[name].fit(X=X, y=Y, w=W)
                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X.shape[1])),
                    explainer_list=explainer_list,
                )
                learner_explanations[name] = learner_explainers[name].explain(X[:limit])

            for explainer_name in explainer_list:
                attribution_pred = pred_explanations[explainer_name]
                attribution_prog = prog_explanations[explainer_name]
                for learner_name in learners:
                    attribution_est = learner_explanations[learner_name][explainer_name]
                    pred_scores = attribution_fractions(
                        attribution_pred, attribution_est
                    )
                    prog_scores = attribution_fractions(
                        attribution_prog, attribution_est
                    )
                    explainability_data.append(
                        [
                            prognostic_mask,
                            learner_name,
                            explainer_name,
                            np.mean(pred_scores),
                            np.std(pred_scores),
                            np.mean(prog_scores),
                            np.std(prog_scores),
                        ]
                    )

        metrics_df = pd.DataFrame(
            explainability_data,
            columns=[
                "Prognostic Mask",
                "Learner",
                "Explainer",
                "Pred% avg",
                "Pred% std",
                "Prog% avg",
                "Prog% std",
            ],
        )

        fig_path = self.save_path / "figures/prog_sens/"
        log.info(f"Saving Figures in {fig_path}...")
        if not fig_path.exists():
            fig_path.mkdir(parents=True, exist_ok=True)

        for metric in ["Pred% avg", "Pred% std", "Prog% avg", "Prog% std"]:
            fig = dataframe_line_plot(
                metrics_df,
                "Prognostic Mask",
                metric,
                explainer_list,
                list(learners.keys()),
            )
            fig.savefig(fig_path / f"{metric}-seed{self.seed}.pdf")


class PropensitySensitivity:
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
        prognostic_mask: float = 0.1,
        save_path: Path = Path.cwd(),
        prop_scales: list = [0.1, 0.2, 0.5, 1],
        treatment_assgn: str = "random",
        addvar_scale: float = 0.1,
    ) -> None:

        self.n_units_hidden = n_units_hidden
        self.n_layers = n_layers
        self.penalty_orthogonal = penalty_orthogonal
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.seed = seed
        self.explainer_limit = explainer_limit
        self.save_path = save_path
        self.prop_scales = prop_scales
        self.prognostic_mask = prognostic_mask
        self.treatment_assgn = treatment_assgn
        self.addvar_scale = addvar_scale

    def run(
        self,
        dataset: str = "twins",
        explainer_list: list = ["integrated_gradients"],
        learner_list: list = ["TLearner", "SLearner", "TARNet", "CFRNet", "SNet"],
        run_name: str = "results",
        n_layers_r: int = 1,
        ortho_reg_type: str = "abs",
        penalty_orthogonal: float = 0.01,
    ) -> None:

        X_raw, T_raw, Y_raw, _, _, _ = load(dataset, train_ratio=1.0)

        log.info("Fitting DGP...")
        sim = Simulator(
            X_raw,
            T_raw,
            Y_raw,
            n_units_hidden=self.n_units_hidden,
            n_layers=self.n_layers,
            penalty_orthogonal=self.penalty_orthogonal,
            batch_size=self.batch_size,
            n_iter=self.n_iter,
            seed=self.seed,
        )

        explainability_data = []

        for prop_scale in self.prop_scales:
            log.info(f"Now working with prop_scale = {prop_scale}...")
            X, W, Y, po0, po1, prop, dgp_model, top_idx = sim.simulate_dataset(
                X_raw,
                prognostic_mask=self.prognostic_mask,
                scale_factor=10,
                noise=True,
                err_std=0.1,
                prop_scale=prop_scale,
                treatment_assign=self.treatment_assgn,
                addvar_scale=self.addvar_scale,
            )

            dgp_te = DgpFunctionWrapper(sim, "te")
            dgp_prog = DgpFunctionWrapper(sim, "prog")

            log.info("Explaining DGP...")
            limit = self.explainer_limit
            # Generate the feature importance for the prognostic model
            prog_explainer = Explainer(
                dgp_prog,
                feature_names=list(range(X.shape[1])),
                explainer_list=explainer_list,
            )
            prog_explanations = prog_explainer.explain(X[:limit])
            # Generate the feature importance for the predictive model(po1- po0 importance)
            pred_explainer = Explainer(
                dgp_te,
                feature_names=list(range(X.shape[1])),
                explainer_list=explainer_list,
            )
            pred_explanations = pred_explainer.explain(X[:limit])

            log.info("Fitting and explaining learners...")
            all_learners = {
                "TLearner": cate_models.torch.TLearner(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_out=n_layers_r + 1,
                    n_units_out=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "SLearner": cate_models.torch.SLearner(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_out=n_layers_r + 1,
                    n_units_out=100,
                    n_iter=self.n_iter,
                    batch_size=1024,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "TARNet": cate_models.torch.TARNet(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_r=n_layers_r,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                ),
                "CFRNet": cate_models.torch.TARNet(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_r=n_layers_r,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                    penalty_disc=0.01,
                ),
                "CFRNet_0.001": cate_models.torch.TARNet(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_r=n_layers_r,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                    penalty_disc=0.001,
                ),
                "CFRNet_0.0001": cate_models.torch.TARNet(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_r=n_layers_r,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=100,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    nonlin="relu",
                    penalty_disc=0.0001,
                ),
                "SNet": cate_models.torch.SNet(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_r=n_layers_r,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=50,
                    n_units_r_small=50,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    penalty_orthogonal=penalty_orthogonal,
                    nonlin="relu",
                    ortho_reg_type=ortho_reg_type,
                ),
                "SNet_noprop": cate_models.torch.SNet(
                    X.shape[1],
                    binary_y=(len(np.unique(Y)) == 2),
                    n_layers_r=n_layers_r,
                    n_layers_out=1,
                    n_units_out=100,
                    n_units_r=50,
                    n_units_r_small=50,
                    batch_size=1024,
                    n_iter=self.n_iter,
                    batch_norm=False,
                    penalty_orthogonal=penalty_orthogonal,
                    nonlin="relu",
                    with_prop=False,
                    ortho_reg_type=ortho_reg_type,
                ),
            }

            learners = {name: all_learners[name] for name in learner_list}

            learner_explainers = {}
            learner_explanations = {}
            learner_pehe = {}
            learner_factual_rmse = {}

            for name in learners:
                log.info(f"Training learner {name}")
                learners[name].fit(X=X, y=Y, w=W)
                learner_explainers[name] = Explainer(
                    learners[name],
                    feature_names=list(range(X.shape[1])),
                    explainer_list=explainer_list,
                )
                learner_explanations[name] = learner_explainers[name].explain(X[:limit])

                # also create PEHE and factual prediction
                _, mu0_pred, mu1_pred = learners[name].predict(X=X, return_po=True)
                pehe, factual_rmse = compute_cate_metrics(
                    po1 - po0, Y, W, mu0_pred, mu1_pred
                )
                learner_pehe[name] = pehe
                learner_factual_rmse[name] = factual_rmse

            for explainer_name in explainer_list:
                attribution_pred = pred_explanations[explainer_name]
                attribution_prog = prog_explanations[explainer_name]
                for learner_name in learners:
                    attribution_est = learner_explanations[learner_name][explainer_name]
                    pred_scores = attribution_fractions(
                        attribution_pred, attribution_est
                    )
                    prog_scores = attribution_fractions(
                        attribution_prog, attribution_est
                    )

                    if top_idx is not None:
                        # calculate index score
                        idx_score = attribution_fractions(
                            attribution_pred, attribution_est, top_idx=top_idx
                        )
                        explainability_data.append(
                            [
                                self.seed,
                                self.prognostic_mask,
                                self.treatment_assgn,
                                prop_scale,
                                learner_name,
                                explainer_name,
                                np.mean(pred_scores),
                                np.std(pred_scores),
                                np.mean(prog_scores),
                                np.std(prog_scores),
                                np.mean(idx_score),
                                np.std(idx_score),
                                learner_pehe[learner_name],
                                learner_factual_rmse[learner_name],
                            ]
                        )
                    else:
                        explainability_data.append(
                            [
                                self.seed,
                                self.prognostic_mask,
                                self.treatment_assgn,
                                prop_scale,
                                learner_name,
                                explainer_name,
                                np.mean(pred_scores),
                                np.std(pred_scores),
                                np.mean(prog_scores),
                                np.std(prog_scores),
                                np.nan,
                                np.nan,
                                learner_pehe[learner_name],
                                learner_factual_rmse[learner_name],
                            ]
                        )

                if top_idx is not None:
                    # calculate index score
                    idxtruth_score = attribution_fractions(
                        attribution_pred, attribution_pred, top_idx=top_idx
                    )

                    explainability_data.append(
                        [
                            self.seed,
                            self.prognostic_mask,
                            self.treatment_assgn,
                            prop_scale,
                            "Truth",
                            explainer_name,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.nan,
                            np.mean(idxtruth_score),
                            np.std(idxtruth_score),
                            np.nan,
                            np.nan,
                        ]
                    )

        df_columns = [
            "Random Seed",
            "Prognostic Mask",
            "Treament Assignement",
            "Propensity Scale",
            "Learner",
            "Explainer",
            "Pred% avg",
            "Pred% std",
            "Prog% avg",
            "Prog% std",
            "Top Idx% avg",
            "Top Idx% std",
            "PEHE",
            "Factual RMSE",
        ]
        metrics_df = pd.DataFrame(
            explainability_data,
            columns=df_columns,
        )

        # Create raw results directories
        res_path = self.save_path / "results/prop_sens/"
        log.info(f"Saving Data in {res_path}...")
        if not res_path.exists():
            res_path.mkdir(parents=True, exist_ok=True)

        csv_path = res_path / f"{run_name}.csv"
        # Add the header if the csv file does not exist yet
        if not csv_path.exists():
            with open(csv_path, "w", encoding="UTF8") as f:
                writer = csv.writer(f)
                writer.writerow(df_columns)
        # Append th dataframe to the csv
        metrics_df.to_csv(csv_path, mode="a", header=False, index=False)

        # Save figures
        fig_path = self.save_path / "figures/prop_sens/"
        log.info(f"Saving Figures in {fig_path}...")
        if not fig_path.exists():
            fig_path.mkdir(parents=True, exist_ok=True)

        for metric in [
            "Pred% avg",
            "Prog% avg",
            "Top Idx% avg",
            "PEHE",
            "Factual RMSE",
        ]:
            if metric == "Top Idx% avg" and top_idx is None:
                continue
            fig = dataframe_line_plot(
                metrics_df,
                "Propensity Scale",
                metric,
                explainer_list,
                list(learners.keys()) + ["Truth"],
                x_logscale=False,
            )
            fig.savefig(
                fig_path
                / (
                    run_name + f"-{metric}-Seed={self.seed}"
                    f"-Progmask={self.prognostic_mask}"
                    f"-Treatment_assgn={self.treatment_assgn}.pdf"
                ),
            )
            plt.close(fig)
