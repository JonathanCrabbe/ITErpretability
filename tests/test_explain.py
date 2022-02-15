import numpy as np
import pytest
from catenets.datasets import load
from catenets.models.torch import TLearner

from iterpretability.explain import Explainer

explainer_list = [
    "feature_ablation",
    "integrated_gradients",
    "deeplift",
    "feature_permutation",
    "lime",
    "shapley_value_sampling",
    "kernel_shap",
]


@pytest.mark.parametrize("dataset", ["ihdp"])
def test_explainer(dataset: str) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()
    Y_train = Y_train.ravel()

    learner = TLearner(
        X_train.shape[1], binary_y=(len(np.unique(Y_train)) == 2), n_iter=100
    )
    learner.fit(X=X_train, y=Y_train, w=W_train)
    explainer = Explainer(
        learner,
        feature_names=list(range(X_train.shape[1])),
        n_samples=100,
        n_steps=100,
        explainer_list=explainer_list,
    )

    output = explainer.explain(X_test)

    for method in output:
        assert output[method].shape == X_test.shape
        assert output[method].sum() != 0


@pytest.mark.parametrize("dataset", ["ihdp"])
def test_plot(dataset: str) -> None:
    X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load(dataset)
    W_train = W_train.ravel()
    Y_train = Y_train.ravel()

    learner = TLearner(
        X_train.shape[1], binary_y=(len(np.unique(Y_train)) == 2), n_iter=100
    )
    learner.fit(X=X_train, y=Y_train, w=W_train)
    explainer = Explainer(
        learner,
        feature_names=list(range(X_train.shape[1])),
        n_samples=100,
        n_steps=100,
        explainer_list=explainer_list,
    )

    explainer.plot(X_test)
