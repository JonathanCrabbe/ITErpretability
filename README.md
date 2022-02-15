# ITErpretability
Explainability in the CATE setting


## Installation

```bash
pip install -r requirements.txt
pip install .
```

## Tests

You can run the tests using

```bash
pip install -r requirements_dev.txt

pytest -vsx
```

## Usage examples

### 1. Generate a new dataset starting from the Twins dataset

```python
from catenets.datasets import load
from iterpretability import Simulator

X_raw, T_raw, Y_raw, _, _, _ = load("twins")

sim = Simulator(X_raw, T_raw, Y_raw, n_iter=10)

X, W, Y, prog, po0, po1, _ = sim.simulate_dataset(X_raw)

```
### 2. Generate feature importance for a learner

```python
import numpy as np
from catenets.datasets import load
from catenets.models.torch import TLearner

from iterpretability import Explainer

X_train, W_train, Y_train, Y_train_full, X_test, Y_test = load("twins")
W_train = W_train.ravel()
Y_train = Y_train.ravel()

learner = TLearner(
    X_train.shape[1], binary_y=(len(np.unique(Y_train)) == 2), n_iter=100
)
learner.fit(X=X_train, y=Y_train, w=W_train)
explainer = Explainer(learner, feature_names=list(range(X_train.shape[1])))

explanations = explainer.explain(X_test[:2])
explanations
```

### 3. Hyperparameter search for the DGP

```python
from catenets.datasets import load
from iterpretability.hyperparam_search import search

X_raw, T_raw, Y_raw, Y_full_raw, _, _ = load("ihdp")
best_params = search(X_raw, T_raw, Y_raw, Y_full_raw, n_trials=5)

best_params
```
### 4. Generate prognostic, predictive and learner explanations

```python
import numpy as np

from catenets.datasets import load
import catenets.models as cate_models

from iterpretability import Experiment

X_raw, T_raw, Y_raw, Y_full_raw, _, _ = load("ihdp")

learners = {
    "SLearner": cate_models.torch.SLearner(
        X_raw.shape[1],
        binary_y=(len(np.unique(Y_raw)) == 2),
        nonlin="selu",
        n_iter=100,
    )
}
ctx = Experiment(learners)

(
    prognostic_explanations,
    predictive_explanations,
    learner_explanations,
    learner_po_explanations,
) = ctx.run(X_raw, T_raw, Y_raw, explainer_limit=2, n_iter=10)
```
