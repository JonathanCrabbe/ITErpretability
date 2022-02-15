import pytest
from catenets.datasets import load

from iterpretability.hyperparam_search import search


@pytest.mark.parametrize("dataset", ["ihdp"])
def test_search(dataset: str) -> None:
    X_raw, T_raw, Y_raw, Y_full_raw, _, _ = load(dataset)

    best_params = search(X_raw, T_raw, Y_raw, Y_full_raw, n_trials=2)

    for key in [
        "batch_size",
        "n_units_hidden",
        "n_layers",
        "lr",
        "weight_decay",
        "n_iter",
    ]:
        assert key in best_params
