from typing import Any

import numpy as np
import pytest
from catenets.datasets import load
from catenets.experiment_utils.tester import evaluate_treatments_model
from catenets.models.torch import SLearner, TLearner

import iterpretability.logger as log
from iterpretability.simulate import Simulator


@pytest.mark.parametrize("dataset", ["ihdp", "twins"])
@pytest.mark.parametrize("model_t", [TLearner, SLearner])
def test_learner_sanity(dataset: str, model_t: Any) -> None:
    X_raw, T_raw, Y_raw, Y_full_raw, _, _ = load(dataset)

    sim = Simulator(X_raw, T_raw, Y_raw, n_iter=50)
    X, W, Y, po0, po1, _, _, _ = sim.simulate_dataset(X_raw)
    Y_full = np.asarray([po0, po1]).squeeze().T

    model = model_t(X.shape[1], binary_y=(len(np.unique(Y)) == 2), n_iter=50)
    score = evaluate_treatments_model(model, X, Y, Y_full, W)

    log.info(model_t, score)
