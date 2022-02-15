from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum.attr import (
    DeepLift,
    FeatureAblation,
    FeaturePermutation,
    IntegratedGradients,
    KernelShap,
    Lime,
    ShapleyValueSampling,
)
from captum.attr._core.lime import get_exp_kernel_similarity_function
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Explainer:
    """
    Explainer instance, consisting of several explainability methods.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List,
        explainer_list: List = [
            "feature_ablation",
            "integrated_gradients",
            "deeplift",
            "feature_permutation",
            "lime",
            "shapley_value_sampling",
            "kernel_shap",
        ],
        n_steps: int = 500,
        perturbations_per_eval: int = 10,
        n_samples: int = 1000,
        kernel_width: float = 1.0,
    ) -> None:
        # Feature ablation
        feature_ablation_model = FeatureAblation(model)

        def normalize(X: torch.Tensor) -> torch.Tensor:
            XX = X.view(X.size(0), -1)
            XX -= XX.min(1, keepdim=True)[0]
            XX /= XX.max(1, keepdim=True)[0] + 1e-8
            return XX.view(X.shape)

        def feature_ablation_cbk(X_test: torch.Tensor) -> torch.Tensor:
            out = feature_ablation_model.attribute(
                X_test, n_steps=n_steps, perturbations_per_eval=perturbations_per_eval
            )

            return out

        # Integrated gradients
        integrated_gradients_model = IntegratedGradients(model)

        def integrated_gradients_cbk(X_test: torch.Tensor) -> torch.Tensor:
            return integrated_gradients_model.attribute(X_test, n_steps=n_steps)

        # DeepLift
        deeplift_model = DeepLift(model)

        def deeplift_cbk(X_test: torch.Tensor) -> torch.Tensor:
            return deeplift_model.attribute(X_test)

        # Feature permutation
        feature_permutation_model = FeaturePermutation(model)

        def feature_permutation_cbk(X_test: torch.Tensor) -> torch.Tensor:
            return feature_permutation_model.attribute(
                X_test, n_steps=n_steps, perturbations_per_eval=perturbations_per_eval
            )

        # LIME
        exp_eucl_distance = get_exp_kernel_similarity_function(
            kernel_width=kernel_width
        )
        lime_model = Lime(
            model,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=exp_eucl_distance,
        )

        def lime_cbk(X_test: torch.Tensor) -> torch.Tensor:
            return lime_model.attribute(
                X_test,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
            )

        # Shapley value sampling
        shapley_value_sampling_model = ShapleyValueSampling(model)

        def shapley_value_sampling_cbk(X_test: torch.Tensor) -> torch.Tensor:
            return shapley_value_sampling_model.attribute(
                X_test,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
            )

        # Kernel SHAP
        kernel_shap_model = KernelShap(model)

        def kernel_shap_cbk(X_test: torch.Tensor) -> torch.Tensor:
            return kernel_shap_model.attribute(
                X_test,
                n_samples=n_samples,
                perturbations_per_eval=perturbations_per_eval,
            )

        self.explainers = {
            "feature_ablation": feature_ablation_cbk,
            "integrated_gradients": integrated_gradients_cbk,
            "deeplift": deeplift_cbk,
            "feature_permutation": feature_permutation_cbk,
            "lime": lime_cbk,
            "shapley_value_sampling": shapley_value_sampling_cbk,
            "kernel_shap": kernel_shap_cbk,
        }
        self.explainer_list = explainer_list
        self.feature_names = feature_names

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).float().to(DEVICE)

    def explain(self, X: torch.Tensor) -> Dict:
        output = {}
        for name in self.explainer_list:
            X_test = self._check_tensor(X)
            X_test.requires_grad_()

            explainer = self.explainers[name]
            output[name] = explainer(X_test).detach().cpu().numpy()

        return output

    def plot(self, X: torch.Tensor) -> None:
        explanations = self.explain(X)

        fig, axs = plt.subplots(int((len(explanations) + 1) / 2), 2)

        idx = 0
        for name in explanations:
            x_pos = np.arange(len(self.feature_names))

            ax = axs[int(idx / 2), idx % 2]

            ax.bar(x_pos, np.mean(explanations[name], axis=0), align="center")
            ax.set_xlabel("Features")
            ax.set_title(f"{name}")

            idx += 1
