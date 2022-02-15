from pathlib import Path

import catenets.models as cate_models
import numpy as np
import pytest
from catenets.datasets import load

from iterpretability.experiment import (
    Experiment,
    PrognosticSensitivity,
    PropensitySensitivity,
)


@pytest.mark.parametrize("dataset", ["ihdp"])
def test_search(dataset: str) -> None:
    X_raw, T_raw, Y_raw, Y_full_raw, _, _ = load(dataset)

    exp = Experiment(
        {
            "SLearner": cate_models.torch.SLearner(
                X_raw.shape[1],
                binary_y=(len(np.unique(Y_raw)) == 2),
                nonlin="selu",
                n_iter=100,
            )
        }
    )
    prog_explanations, pred_explanations, explanations, po_explanations = exp.run(
        X_raw, T_raw, Y_raw, explainer_limit=2, n_iter=10
    )

    dgp_outputs = [prog_explanations, pred_explanations]
    learner_output = [explanations, po_explanations]

    for out in dgp_outputs:
        assert len(out) == 7
        for explainer in out:
            assert len(out[explainer]) == 2

    for out in learner_output:
        assert len(out) == 1
        assert "SLearner" in out
        if isinstance(out["SLearner"], tuple):
            assert len(out["SLearner"][0]) == 7
            assert len(out["SLearner"][1]) == 7
        else:
            assert len(out["SLearner"]) == 7


def test_prognostic_sensitivity() -> None:
    save_folder = Path(__file__).resolve().parent
    experiment = PrognosticSensitivity(
        seed=42,
        explainer_limit=2,
        n_iter=1,
        prognostic_masks=[1e-3],
        save_path=save_folder,
    )
    experiment.run(
        explainer_list=[
            "integrated_gradients",
            "shapley_value_sampling",
            "feature_permutation",
            "feature_ablation",
        ]
    )
    assert (save_folder / "figures/prog_sens").exists()


def test_propensity_sensitivity() -> None:
    save_folder = Path(__file__).resolve().parent
    experiment = PropensitySensitivity(
        seed=42,
        explainer_limit=2,
        n_iter=1,
        save_path=save_folder,
    )
    experiment.run(explainer_list=["integrated_gradients"])
    assert (save_folder / "figures/prop_sens").exists()
    assert (save_folder / "results/prop_sens").exists()
