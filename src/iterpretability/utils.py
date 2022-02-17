# stdlib
import random
from typing import Optional

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
from torch import nn

import iterpretability.simulate

abbrev_dict = {
    "shapley_value_sampling": "SVS",
    "integrated_gradients": "IG",
    "kernel_shap": "SHAP",
    "gradient_shap": "GSHAP",
    "feature_permutation": "FP",
    "feature_ablation": "FA",
    "deeplift": "DL",
    "lime": "LIME",
}

explainer_symbols = {
    "shapley_value_sampling": "D",
    "integrated_gradients": "8",
    "kernel_shap": "s",
    "feature_permutation": "<",
    "feature_ablation": "x",
    "deeplift": "H",
    "lime": ">",
}

cblind_palete = sns.color_palette("colorblind", as_cmap=True)
learner_colors = {
    "SLearner": cblind_palete[0],
    "TLearner": cblind_palete[1],
    "SNet": cblind_palete[2],
    "TARNet": cblind_palete[3],
    "CFRNet": cblind_palete[4],
    "SNet_noprop": cblind_palete[5],
    "CFRNet_0.001": cblind_palete[6],
    "CFRNet_0.0001": cblind_palete[7],
    "Truth": cblind_palete[7],
}


def enable_reproducible_results(seed: int = 42) -> None:
    """
    Set a fixed seed for all the used libraries

    Args:
        seed: int
            The seed to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class PotentialOutcomeWrapper(nn.Module):
    """
    Wrapper over the CATENets models to return one of the potential outcomes instead of the treatment effects
    """

    def __init__(self, model: nn.Module, po: int = 0) -> None:
        super(PotentialOutcomeWrapper, self).__init__()

        self.po = po
        self.model = model

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        te, po0, po1 = self.model.predict(X, return_po=True)

        return po1 if self.po else po0


class DgpFunctionWrapper(nn.Module):
    """
    Wrapper over the DGP to return the treatment effect / potential outcome or prognostic effect.
    """

    def __init__(self, dgp: "iterpretability.simulate.Simulator", output: str) -> None:
        super(DgpFunctionWrapper, self).__init__()
        if output not in ["te", "po0", "po1", "prog"]:
            raise ValueError(
                "The output should either be the treatment effect (te),"
                " the prognostic part (prog) or one of the potential outcomes (po0 or po1)"
            )
        self.dgp = dgp
        self.output = output

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.output == "te":
            res = self.dgp.te(X)
        elif self.output == "po0":
            res = self.dgp.po0(X)
        elif self.output == "po1":
            res = self.dgp.po1(X)
        else:
            res = self.dgp.prog(X)
        return res


def attribution_fractions(
        attribution_true: np.ndarray,
        attribution_est: np.ndarray,
        detection_tresh: float = 0,
        top_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Compute the fraction of attribution_est that is truly important for attribution_true for each example
    """

    # Set the detection threshold to a default value if unspecified
    if detection_tresh == 0:
        detection_tresh = 1 / attribution_true.shape[-1]
    elif detection_tresh < 0 or detection_tresh > 1:
        raise ValueError("The detection threshold should be between 0 and 1")

    # Create a mask highlighting the important features in the true array
    if top_idx is None:
        attribution_true = np.abs(attribution_true)
        attribution_true = attribution_true / (
            np.sum(attribution_true, axis=-1, keepdims=True)
        )
        mask = attribution_true >= detection_tresh
    else:
        mask = np.zeros(shape=attribution_est.shape)
        mask[:, top_idx] = 1

    # Compute the portion of the estimated attribution highlighted by the mask
    attribution_est = np.abs(attribution_est)
    attribution_est_masked = mask * attribution_est
    return np.sum(attribution_est_masked, axis=-1) / (np.sum(attribution_est, axis=-1))


def dgp_importance_plot(
        pred_scores: np.ndarray,
        prog_scores: np.ndarray,
        po0_scores: np.ndarray,
        po1_scores: np.ndarray,
        explainer_name: str,
) -> plt.Figure:
    sns.set_style("white")
    fig, axs = plt.subplots(1, 3, figsize=(25, 5))
    color_palette = sns.blend_palette(
        ["black", "red", "yellow", "green", "blue", "magenta", "pink"],
        n_colors=pred_scores.shape[1],
    )

    # Make the Predictive and Prognostic importance plots
    sns.boxplot(data=prog_scores, ax=axs[0], palette=color_palette)
    axs[0].set_title(f"Prognostic Importance - {explainer_name}")
    axs[0].set_xlabel("Covariate Number")
    axs[0].set_ylabel("Prognostic Importance")
    sns.boxplot(data=pred_scores, ax=axs[1], palette=color_palette)
    axs[1].set_title(f"Predictive Importance - {explainer_name}")
    axs[1].set_xlabel("Covariate Number")
    axs[1].set_ylabel("Predictive Importance")

    # Hide some of the ticks
    for ind, label in enumerate(axs[0].get_xticklabels()):
        if ind % 5 == 0:  # every 5th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    for ind, label in enumerate(axs[1].get_xticklabels()):
        if ind % 5 == 0:  # every 5th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    # Put the PO importance scores in a dataframe
    po_list = []
    for covariate in range(pred_scores.shape[1]):
        for example in range(pred_scores.shape[0]):
            po0_covariate_score = po0_scores[example, covariate]
            po1_covariate_score = po1_scores[example, covariate]
            po_list.append([po0_covariate_score, po1_covariate_score, covariate])
    po_df = pd.DataFrame(po_list, columns=["PO0_Score", "PO1_Score", "Covariate"])

    # Make the PO importance plots
    sns.scatterplot(
        data=po_df,
        x="PO0_Score",
        y="PO1_Score",
        hue="Covariate",
        ax=axs[2],
        palette=color_palette,
        alpha=0.1,
    )
    axs[2].get_legend().remove()
    x_vals = np.array(axs[2].get_xlim())
    axs[2].plot(x_vals, x_vals, "--")
    axs[2].set_title(f"Potential Outcomes Importance - {explainer_name}")
    axs[2].set_xlabel("PO_0 Importance")
    axs[2].set_ylabel("PO_1 Importance")
    return fig


def dataframe_line_plot(
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        explainers: list,
        learners: list,
        x_logscale: bool = True,
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.set_style("white")
    for learner_name in learners:
        for explainer_name in explainers:
            sub_df = df.loc[
                (df["Learner"] == learner_name) & (df["Explainer"] == explainer_name)
                ]
            mask_values = sub_df.loc[:, x_axis].values
            metric_values = sub_df.loc[:, y_axis].values
            ax.plot(
                mask_values,
                metric_values,
                color=learner_colors[learner_name],
                marker=explainer_symbols[explainer_name],
            )

    learner_lines = [
        Line2D([0], [0], color=learner_colors[learner_name], lw=2)
        for learner_name in learners
    ]
    explainer_lines = [
        Line2D([0], [0], color="black", marker=explainer_symbols[explainer_name])
        for explainer_name in explainers
    ]

    legend_learners = plt.legend(
        learner_lines, learners, loc="lower left", bbox_to_anchor=(1.04, 0.7)
    )
    legend_explainers = plt.legend(
        explainer_lines,
        [abbrev_dict[explainer_name] for explainer_name in explainers],
        loc="lower left",
        bbox_to_anchor=(1.04, 0),
    )
    plt.subplots_adjust(right=0.75)
    ax.add_artist(legend_learners)
    ax.add_artist(legend_explainers)
    if x_logscale:
        ax.set_xscale("log")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return fig


def compute_cate_metrics(
        cate_true: np.ndarray,
        y_true: np.ndarray,
        w_true: np.ndarray,
        mu0_pred: torch.Tensor,
        mu1_pred: torch.Tensor,
) -> tuple:
    mu0_pred = mu0_pred.detach().cpu().numpy()
    mu1_pred = mu1_pred.detach().cpu().numpy()

    cate_pred = mu1_pred - mu0_pred

    pehe = np.sqrt(mean_squared_error(cate_true, cate_pred))

    y_pred = w_true.reshape(len(cate_true), ) * mu1_pred.reshape(len(cate_true), ) + (
            1
            - w_true.reshape(
        len(cate_true),
    )
    ) * mu0_pred.reshape(
        len(cate_true),
    )
    factual_rmse = np.sqrt(
        mean_squared_error(
            y_true.reshape(
                len(cate_true),
            ),
            y_pred,
        )
    )
    return pehe, factual_rmse


def attribution_accuracy(target_features: list, feature_attributions: np.ndarray) -> float:
    """
 Computes the fraction of the most important features that are truly important
 Args:
     target_features: list of truly important feature indices
     feature_attributions: feature attribution outputted by a feature importance method

 Returns:
     Fraction of the most important features that are truly important
    """

    n_important = len(target_features)  # Number of features that are important
    largest_attribution_idx = torch.topk(torch.from_numpy(feature_attributions), n_important)[1] # Features with largest attribution
    accuracy = 0  # Attribution accuracy
    for k in range(len(largest_attribution_idx)):
        accuracy += len(np.intersect1d(largest_attribution_idx[k], target_features))
    return accuracy / (len(feature_attributions)*n_important)
