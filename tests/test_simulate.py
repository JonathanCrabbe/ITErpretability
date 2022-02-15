import pytest
from catenets.datasets import load

from iterpretability.simulate import BasicNet, RawEstimator, Simulator


def test_basic_net() -> None:
    test = BasicNet(
        name="test",
        n_unit_in=2,
        n_layers=3,
        n_units_hidden=5,
        binary_y=True,
        seed=22,
        dropout=0.1,
        tau=0.2,
    )

    assert len(test.model) == 2 + 3 * (1 + 1 + 1) + 1 + 1
    # - input: linear + activation
    # - "n_layers" hidden layers * (dropout + linear + activation)
    # - output layer + activation


def test_random_estimator() -> None:
    test = RawEstimator(
        2,
        seed=2,
        dropout=0.6,
        batch_size=100,
        n_units_hidden=200,
        n_layers=12,
        lr=1e-4,
        weight_decay=1e-5,
        n_iter=200,
        binary_y=True,
    )

    assert test.prog is not None
    assert test.pred0 is not None
    assert test.pred1 is not None
    assert test.binary_y is True
    assert test.n_iter == 200
    assert test.batch_size == 100


@pytest.mark.parametrize("dataset", ["ihdp", "twins"])
def test_dataset_simulation_sanity(dataset: str) -> None:
    X_raw, T_raw, Y_raw, _, _, _ = load(dataset)
    sim = Simulator(X_raw, T_raw, Y_raw, n_iter=10)
    X, W, Y, po0, po1, _, _, _ = sim.simulate_dataset(X_raw)

    assert po0.shape == Y.shape
    assert po1.shape == Y.shape
    assert W.shape == Y.shape
    assert X.shape == X_raw.shape
