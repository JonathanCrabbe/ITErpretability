import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error

import iterpretability.logger as log
from iterpretability.simulate import Simulator


def search(
    X_raw: pd.DataFrame,
    T_raw: pd.DataFrame,
    Y_raw: pd.DataFrame,
    Y_full_raw: pd.DataFrame,
    n_trials: int = 100,
) -> dict:
    """
    Helper for hyperparam tuning for the DGP.

    Args:
        X_raw:
            baseline covariates dataframe.
        T_raw:
            baseline treatments.
        Y_raw:
            baseline outcomes.
        Y_full_raw:
            baseline full outcomes, used for evaluation.
        n_trials:
            number of optimization trials.
    """

    def objective(trial: optuna.Trial) -> None:
        param_space = {
            "batch_size": trial.suggest_categorical("batch_size", [50, 200, 500]),
            "n_units_hidden": trial.suggest_int("n_units_hidden", 10, 200, 10),
            "n_layers": trial.suggest_int("n_layers", 1, 2),
            "lr": trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4, 2e-3, 2e-4]),
            "weight_decay": trial.suggest_categorical(
                "weight_decay", [1e-2, 1e-3, 1e-4, 2e-3, 2e-4]
            ),
            "n_iter": trial.suggest_int("n_iter", 100, 1000, 100),
            "penalty_orthogonal": trial.suggest_categorical(
                "penalty_orthogonal", [1e-2, 1e-3, 1e-1]
            ),
        }
        log.info(f"Trial {trial.number+1}: {param_space} ")
        sim = Simulator(X_raw, T_raw, Y_raw, **param_space)
        X, W, Y, po0, po1, _, _, _ = sim.simulate_dataset(X_raw)
        Y_full = np.asarray([po0, po1]).squeeze().T
        result = mean_squared_error(Y_full_raw, Y_full)
        log.info(f"DGP PO MSE = {result}")
        return result

    study = optuna.create_study(direction="minimize", study_name="dgp_search")
    study.optimize(objective, n_trials=n_trials)

    log.info(f"Number of finished trials: {len(study.trials)}")
    log.info(f"Best trial:  {study.best_trial.params}")

    return study.best_trial.params


"""

    Twins Best trial:  {'batch_size': 50, 'n_units_hidden': 200, 'n_layers': 2,
     'lr': 0.001, 'weight_decay': 0.01, 'n_iter': 100, 'penalty_orthogonal': 0.001}

"""
