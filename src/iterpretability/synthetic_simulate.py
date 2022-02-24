# stdlib
from typing import Tuple
from functools import reduce

# third party
import numpy as np
import torch
from scipy.special import expit
from scipy.stats import zscore

from src.iterpretability.utils import enable_reproducible_results
from src.iterpretability.utils import get_correlated_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 0


class SyntheticSimulatorBase:
    """
    Data generation process.

    Args:
        X: np.ndarray/pd.DataFrame
            Baseline covariates
        seed: int
            Random seed
        predictive_scale: float
            Parameter to control the magnitude of the predictive effect.

    Returns:
        X: the same set of covariates
        W_synth: simulated treatments
        Y_synth: simulated outcomes
        prog_out: the prognostic outcome for X
        po0, po1: potential outcomes for X
        est: the RandomEstimator used
    """

    def __init__(
        self,
        seed: int = 42,
    ) -> None:

        enable_reproducible_results(seed=seed)
        self.seed = seed

        self.prog_mask, self.pred0_mask, self.pred1_mask = None, None, None
        self.prog_weights, self.pred0_weights, self.pred1_weights = None, None, None

    def get_important_features(self, X: np.ndarray, num_important_features: int) -> Tuple:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple:
        raise NotImplementedError

    def simulate_dataset(
        self,
        X: np.ndarray,
        predictive_scale: float = 1,
        scale_factor: float = 1,
        treatment_assign: str = "random",
        noise: bool = False,
        err_std: float = 0.01,
        prop_scale: float = 1,
        addvar_scale: float = 0.1,
        binary_outcome: bool = False
    ) -> Tuple:
        enable_reproducible_results(self.seed)
        self.scale_factor = scale_factor
        self.predictive_scale = predictive_scale
        self.treatment_assign = treatment_assign
        self.noise = noise
        self.err_std = err_std
        self.prop_scale = prop_scale
        self.addvar_scale = addvar_scale

        prog, pred0, pred1 = self.predict(X)

        self.top_idx = None
        if self.treatment_assign == "random":
            # randomly assign treatments
            propensity = 0.5 * self.prop_scale * np.ones(len(X))
        elif self.treatment_assign == "prog":
            # assign treatments according to prognostic score ('self-selection')
            # compute normalized prognostic score to get propensity score
            prog_score = zscore(prog)
            propensity = expit(self.prop_scale * prog_score)
        elif self.treatment_assign == "pred":
            # assign treatments according to predictive score ('doctor knows CATE')
            # compute normalized predictive score to get propensity score
            pred_score = zscore(pred1 - pred0)
            propensity = expit(self.prop_scale * pred_score)
        elif self.treatment_assign == "linear":
            # assign treatment according to random linear predictor as in GANITE
            # simulate coefficient
            coef = np.random.uniform(-0.01, 0.01, size=[X.shape[1], 1])
            exponent = zscore(np.matmul(X, coef)).reshape(
                X.shape[0],
            )
            propensity = expit(self.prop_scale * exponent)
        elif self.treatment_assign == "top_pred":
            # find top predictive feature, assign treatment according to that
            pred_act = (
                (
                    torch.sum(torch.abs(self.est.pred0.model[0].weight), dim=0)
                    + torch.sum(torch.abs(self.est.pred1.model[0].weight), dim=0)
                )
                    .detach()
                    .cpu()
                    .numpy()
            )
            top_idx = np.argmax(pred_act)
            self.top_idx = top_idx
            exponent = zscore(X[:, top_idx]).reshape(
                X.shape[0],
            )
            propensity = expit(self.prop_scale * exponent)
        elif self.treatment_assign == "top_prog":
            # find top prognostic feature, assign treatment according to that
            prog_act = (
                torch.sum(torch.abs(self.est.prog.model[0].weight), dim=0)
                    .detach()
                    .cpu()
                    .numpy()
            )
            top_idx = np.argmax(prog_act)
            self.top_idx = top_idx
            exponent = zscore(X[:, top_idx]).reshape(
                X.shape[0],
            )
            propensity = expit(self.prop_scale * exponent)
        elif self.treatment_assign == "irrelevant_var":
            prog_act = (
                torch.sum(torch.abs(self.est.prog.model[0].weight), dim=0)
                    .detach()
                    .cpu()
                    .numpy()
            )
            top_idx = np.argmax(prog_act)
            self.top_idx = top_idx

            exponent = zscore(X[:, top_idx]).reshape(
                X.shape[0],
            )
            propensity = expit(self.prop_scale * exponent)

            # remove effect of this variable and recompute
            X_local = X.copy()
            X_local[:, top_idx] = 0
            _, _, prog, pred0, pred1 = self.est.predict(X_local)
            prog = prog.detach().cpu().numpy()
            pred0 = pred0.detach().cpu().numpy()
            pred1 = pred1.detach().cpu().numpy()
        else:
            raise ValueError(
                f"{treatment_assign} is not a valid treatment assignment mechanism."
            )

        W_synth = np.random.binomial(1, p=propensity)

        po0 = self.scale_factor * (prog + self.predictive_scale * pred0)
        po1 = self.scale_factor * (prog + self.predictive_scale * pred1)

        error = 0
        if self.noise:
            error = (
                torch.empty(len(X))
                    .normal_(std=self.err_std)
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
            )

        Y_synth = W_synth * po1 + (1 - W_synth) * po0 + error

        if binary_outcome:
            Y_prob = expit(2 * (Y_synth - np.mean(Y_synth)) / np.std(Y_synth))
            Y_synth = np.random.binomial(1, Y_prob)

        return X, W_synth, Y_synth, po0, po1, propensity

    def po0(self, X: np.ndarray) -> np.ndarray:
        prog_factor, pred0_factor, _ = self.predict(X)

        po0 = self.scale_factor * (
            prog_factor + self.predictive_scale * pred0_factor)
        return po0

    def po1(self, X: np.ndarray) -> np.ndarray:
        prog_factor, _, pred1_factor = self.predict(X)
        po1 = self.scale_factor * (prog_factor + self.predictive_scale * pred1_factor)
        return po1

    def te(self, X: np.ndarray) -> np.ndarray:

        _, pred0_factor, pred1_factor = self.predict(X)

        te = self.scale_factor * (pred1_factor - pred0_factor)

        return te

    def prog(self, X: np.ndarray) -> np.ndarray:
        prog_factor, _, _ = self.predict(X)

        return self.scale_factor * prog_factor

    def get_all_important_features(self) -> np.ndarray:
        all_important_features = reduce(np.union1d, (
            np.where((self.prog_mask).astype(np.int32) != 0)[0],
            np.where((self.pred0_mask).astype(np.int32) != 0)[0],
            np.where((self.pred1_mask).astype(np.int32) != 0)[0]))

        return all_important_features

    def get_predictive_features(self) -> np.ndarray:
        pred_features = np.union1d(np.where((self.pred0_mask).astype(np.int32) != 0)[0],
                                   np.where((self.pred1_mask).astype(np.int32) != 0)[0])

        return pred_features

    def get_prognostic_features(self) -> np.ndarray:
        prog_features = np.where((self.prog_mask).astype(np.int32) != 0)
        return prog_features


class SyntheticSimulatorLinear(SyntheticSimulatorBase):
    """
    Data generation process.

    Args:
        X: np.ndarray/pd.DataFrame
            Baseline covariates
        seed: int
            Random seed
        predictive_scale: float
            Parameter to control the magnitude of the predictive effect.

    Returns:
        X: the same set of covariates
        W_synth: simulated treatments
        Y_synth: simulated outcomes
        prog_out: the prognostic outcome for X
        po0, po1: potential outcomes for X
        est: the RandomEstimator used
    """

    def __init__(
        self,
        X: np.ndarray,
        num_important_features: int = 10,
        random_feature_selection: bool = False,
        seed: int = 42,
    ) -> None:
        super(SyntheticSimulatorLinear, self).__init__(seed=seed)

        self.prog_mask, self.pred0_mask, self.pred1_mask = self.get_important_features(X, num_important_features,
                                                                                       random_feature_selection)
        self.prog_weights = np.random.uniform(-1, 1, size=(X.shape[1])) * self.prog_mask
        self.pred0_weights = np.random.uniform(-1, 1, size=(X.shape[1])) * self.pred0_mask
        self.pred1_weights = np.random.uniform(-1, 1, size=(X.shape[1])) * self.pred1_mask

    def get_important_features(self, X: np.ndarray, num_important_features: int,
                               random_feature_selection: bool = False) -> Tuple:
        assert num_important_features <= int(X.shape[0] / 3)
        prog_mask = np.zeros(shape=(X.shape[1]))
        pred0_mask = np.zeros(shape=(X.shape[1]))
        pred1_mask = np.zeros(shape=(X.shape[1]))

        if random_feature_selection:
            all_indices = np.array(range(X.shape[1]))
            prog_indices = np.random.choice(all_indices, num_important_features, replace=False)

            pred0_indices = np.random.choice(np.setdiff1d(all_indices, prog_indices), num_important_features,
                                             replace=False)
            pred1_indices = np.random.choice(reduce(np.setdiff1d, (all_indices, prog_indices, pred0_indices)),
                                             num_important_features, replace=False)

            prog_mask[prog_indices] = 1
            pred0_mask[pred0_indices] = 1
            pred1_mask[pred1_indices] = 1
        else:

            prog_mask[:num_important_features] = 1
            pred0_mask[num_important_features:(2 * num_important_features)] = 1
            pred1_mask[(2 * num_important_features):(3 * num_important_features)] = 1

        return prog_mask, pred0_mask, pred1_mask

    def predict(self, X: np.ndarray) -> Tuple:
        prog = np.dot(X, self.prog_weights)
        pred0 = np.dot(X, self.pred0_weights)
        pred1 = np.dot(X, self.pred1_weights)

        return prog, pred0, pred1


class SyntheticSimulatorPairwise(SyntheticSimulatorBase):
    """
    Synthetic Simulator with Pairwise Interactions
    """

    def __init__(
        self,
        X: np.ndarray,
        num_important_features: int = 10,
        num_interactions: int = 1,
        selection_type: str = "random",
        seed: int = 42,
    ) -> None:
        super(SyntheticSimulatorPairwise, self).__init__(seed=seed)
        assert selection_type in {"correlation_most", "correlation_least", "random"}
        self.prog_mask, self.pred0_mask, self.pred1_mask,\
            self.prog_inter_mask, self.pred0_inter_mask, self.pred1_inter_mask = self.get_important_features(X,
                                                                num_important_features,num_interactions, selection_type)
        self.prog_weights = np.random.uniform(-1, 1, size=(X.shape[1])) * self.prog_mask
        self.pred0_weights = np.random.uniform(-1, 1, size=(X.shape[1])) * self.pred0_mask
        self.pred1_weights = np.random.uniform(-1, 1, size=(X.shape[1])) * self.pred1_mask
        self.prog_inter_weights = np.random.uniform(-1, 1, size=(X.shape[1], X.shape[1])) * self.prog_inter_mask
        self.pred0_inter_weights = np.random.uniform(-1, 1, size=(X.shape[1], X.shape[1])) * self.pred0_inter_mask
        self.pred1_inter_weights = np.random.uniform(-1, 1, size=(X.shape[1], X.shape[1])) * self.pred1_inter_mask

    def get_important_features(self, X: np.ndarray, num_important_features: int, num_interactions: int = 1,
                               selection_type: str = "random") -> Tuple:
        assert 3*(num_important_features + 2*num_interactions) <= int(X.shape[0])
        prog_mask = np.zeros(shape=(X.shape[1]))
        pred0_mask = np.zeros(shape=(X.shape[1]))
        pred1_mask = np.zeros(shape=(X.shape[1]))
        prog_inter_mask = np.zeros(shape=(X.shape[1], X.shape[1]))
        pred0_inter_mask = np.zeros(shape=(X.shape[1], X.shape[1]))
        pred1_inter_mask = np.zeros(shape=(X.shape[1], X.shape[1]))
        if selection_type == "random":
            all_indices = np.array(range(X.shape[1]))
            np.random.shuffle(all_indices)
            prog_indices = all_indices[:num_important_features]
            pred0_indices = all_indices[num_important_features:2*num_important_features]
            pred1_indices = all_indices[2*num_important_features:3*num_important_features]
            prog_inter_indices = all_indices[3*num_important_features:3*num_important_features+2*num_interactions]
            pred0_inter_indices = all_indices[3*num_important_features+2*num_interactions:
                                              3*num_important_features+4*num_interactions]
            pred1_inter_indices = all_indices[3*num_important_features+4*num_interactions:
                                              3*num_important_features+6*num_interactions]
        prog_mask[prog_indices] = 1
        pred0_mask[pred0_indices] = 1
        pred1_mask[pred1_indices] = 1
        prog_inter_mask[prog_inter_indices[::2], prog_inter_indices[1::2]] = 1
        pred0_inter_mask[pred0_inter_indices[::2], pred0_inter_indices[1::2]] = 1
        pred1_inter_mask[pred1_inter_indices[::2], pred1_inter_indices[1::2]] = 1
        return prog_mask, pred0_mask, pred1_mask, prog_inter_mask, pred0_inter_mask, pred1_inter_mask

    def predict(self, X: np.ndarray) -> Tuple:
        prog = np.dot(X, self.prog_weights) + np.einsum('be,ef,bf->b', X, self.prog_inter_weights, X)
        pred0 = np.dot(X, self.pred0_weights) + np.einsum('be,ef,bf->b', X, self.pred0_inter_weights, X)
        pred1 = np.dot(X, self.pred1_weights) + np.einsum('be,ef,bf->b', X, self.pred1_inter_weights, X)
        return prog, pred0, pred1
