# stdlib
from typing import Tuple

# third party
import numpy as np
import torch
from catenets.models.torch.utils.model_utils import make_val_split
from scipy.special import expit
from scipy.stats import zscore
from torch import nn

import iterpretability.logger as log
from iterpretability.utils import enable_reproducible_results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 0


class BasicNet(nn.Module):
    """
    Basic neural net for the random estimator

    Args:
        name: str
            Name used for logs
        n_unit_in: int
            Number of input features
        n_layers: int
            Number of hidden layers
        n_units_hidden: int
            Size of the hidden layers
        binary_y: bool
            If true, a Sigmoid activation is added at the end
        seed: int
            Random seed
        dropout: float
            probability of an element to be zeroed
        tau: float
            LeakyReLU parameter
    """

    def __init__(
        self,
        name: str,
        n_unit_in: int,
        n_layers: int = 1,
        n_units_hidden: int = 10,
        binary_y: bool = False,
        seed: int = 42,
        dropout: float = 0,
        tau: float = 1,
    ) -> None:
        super(BasicNet, self).__init__()

        enable_reproducible_results(seed=seed)

        layers = [
            nn.Linear(n_unit_in, n_units_hidden),
            nn.LeakyReLU(negative_slope=1 - tau),
        ]

        for i in range(n_layers):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(n_units_hidden, n_units_hidden),
                    nn.LeakyReLU(negative_slope=1 - tau),
                ]
            )

        # add final layers
        layers.append(nn.Linear(n_units_hidden, 1))
        if binary_y:
            layers.append(nn.Sigmoid())

        # return final architecture
        self.model = nn.Sequential(*layers).to(DEVICE)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X).squeeze()

    def set_tau(self, tau: float) -> None:
        # Check that tau is in the correct range
        if tau > 1 or tau < 0:
            raise ValueError("The parameter tau should be between 0 and 1.")

        # Scan the layers and change tau for Leaky Relus
        for layer in self.model.modules():
            if isinstance(layer, nn.LeakyReLU):
                layer.negative_slope = 1 - tau


class RawEstimator(torch.nn.Module):
    """
    Learner for the DGP.

    Args:
        n_unit_in: int
            Number of input features
        binary_y: bool
            If true, a Sigmoid activation is added at the end
        tau: float
            LeakyReLU parameter
        prognostic_mask: float
            Parameter to control the prognostic/predictive output
        lr: float
            Learning rate
        weight_decay: float
            Weight decay for the optimizer
        n_iter: int
            Number of training iterations
        batch_size: int
            Batch size used for training
        seed: int
            Random seed
        patience: int
            Number of iterations before early stopping
        n_iter_min: int
            Minimum number of iterations before early stopping
        Dropout: float
            probability of an element to be zeroed.
        n_units_hidden: int
            Size of the hidden layers
        n_layers: int
            Number of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        binary_y: bool = False,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        n_iter: int = 10000,
        batch_size: int = 50,
        n_iter_print: int = 50,
        seed: int = 42,
        val_split_prop: float = 0.3,
        patience: int = 10,
        n_iter_min: int = 1000,
        dropout: float = 0,
        n_units_hidden: int = 10,
        n_layers: int = 1,
        penalty_orthogonal: float = 0.01,
        ortho_reg_type: str = "abs",
    ):
        super(RawEstimator, self).__init__()
        enable_reproducible_results(seed)
        self.prog = BasicNet(
            "prog",
            n_unit_in=n_input,
            seed=seed,
            tau=1,
            dropout=dropout,
            binary_y=binary_y,
            n_units_hidden=n_units_hidden,
            n_layers=n_layers,
        )
        self.pred0 = BasicNet(
            "pred0",
            n_unit_in=n_input,
            seed=seed + 1,
            tau=1,
            dropout=dropout,
            binary_y=binary_y,
            n_units_hidden=n_units_hidden,
            n_layers=n_layers,
        )
        self.pred1 = BasicNet(
            "pred1",
            n_unit_in=n_input,
            seed=seed + 1,
            tau=1,
            dropout=dropout,
            binary_y=binary_y,
            n_units_hidden=n_units_hidden,
            n_layers=n_layers,
        )

        self.binary_y = binary_y
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_iter_print = n_iter_print
        self.seed = seed
        self.val_split_prop = val_split_prop
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.penalty_orthogonal = penalty_orthogonal
        self.ortho_reg_type = ortho_reg_type
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X)).float().to(DEVICE)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        po0, po1, _, _, _ = self.predict(X)
        return po1 - po0

    def predict(self, X: torch.Tensor) -> tuple:
        X = self._check_tensor(X)

        self.prog.eval()
        self.pred0.eval()
        self.pred1.eval()

        # Ensure that the predictions are in [0,1] when binary
        probability_factor = 0.5 if self.binary_y else 1

        # Evaluate the output of the three NNs
        prog = self.prog(X)
        pred0 = self.pred0(X)
        pred1 = self.pred1(X)

        # Combine these outputs to estimate the POs
        po0 = probability_factor * (prog + pred0)
        po1 = probability_factor * (prog + pred1)

        return (
            po0.squeeze(),
            po1.squeeze(),
            prog.squeeze(),
            pred0.squeeze(),
            pred1.squeeze(),
        )

    def _ortho_reg(self) -> float:
        def _get_absolute_rowsums(mat: torch) -> torch.Tensor:
            return torch.sum(torch.abs(mat), dim=0)

        def _get_cos_reg(
            params_0: torch.Tensor, params_1: torch.Tensor, normalize: bool = False
        ) -> torch.Tensor:
            x_min = min(params_0.shape[0], params_1.shape[0])
            y_min = min(params_0.shape[1], params_1.shape[1])

            return (
                torch.linalg.norm(
                    params_0[:x_min, :y_min] * params_1[:x_min, :y_min], "fro"
                )
                ** 2
            )

        prog_params = self.prog.model[0].weight
        pred0_params = self.pred0.model[0].weight
        pred1_params = self.pred1.model[0].weight

        if self.ortho_reg_type == "abs":
            return self.penalty_orthogonal * torch.sum(
                _get_absolute_rowsums(prog_params) * _get_absolute_rowsums(pred0_params)
                + _get_absolute_rowsums(prog_params)
                * _get_absolute_rowsums(pred1_params)
                + _get_absolute_rowsums(pred0_params)
                * _get_absolute_rowsums(pred1_params)
            )
        elif self.ortho_reg_type == "fro":
            return self.penalty_orthogonal * (
                _get_cos_reg(prog_params, pred0_params)
                + _get_cos_reg(prog_params, pred1_params)
                + _get_cos_reg(pred1_params, pred0_params)
            )
        else:
            return 0

    def train(
        self, X: torch.Tensor, T: torch.Tensor, y: torch.Tensor
    ) -> "RawEstimator":
        X = self._check_tensor(X)
        y = self._check_tensor(y).squeeze()
        T = self._check_tensor(T).long().squeeze()

        X, y, T, X_val, y_val, T_val, val_string = make_val_split(
            X, y, w=T, val_split_prop=self.val_split_prop, seed=self.seed
        )
        y_val = y_val.squeeze()
        n = X.shape[0]  # could be different from before due to split

        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        # Put the NNs in training mode
        self.prog.train()
        self.pred0.train()
        self.pred1.train()

        # Train
        val_loss_best = 99999
        patience = 0
        for i in range(self.n_iter):
            # shuffle data for minibatches
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                self.optimizer.zero_grad()

                idx_next = train_indices[
                    (b * batch_size) : min((b + 1) * batch_size, n - 1)
                ]

                X_next = X[idx_next].to(DEVICE)
                y_next = y[idx_next].to(DEVICE)
                t_next = T[idx_next].to(DEVICE)

                loss = nn.BCELoss() if self.binary_y else nn.MSELoss()

                po0, _, _, _, _ = self.predict(X_next[t_next == 0])
                batch_loss = loss(po0, y_next[t_next == 0])

                _, po1, _, _, _ = self.predict(X_next[t_next == 1])
                batch_loss += loss(po1, y_next[t_next == 1])
                batch_loss += self._ortho_reg()

                batch_loss.backward()

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if i % self.n_iter_print == 0:
                loss = nn.BCELoss() if self.binary_y else nn.MSELoss()

                self.prog.eval()
                self.pred0.eval()
                self.pred1.eval()

                with torch.no_grad():
                    po0, po1, _, _, _ = self.predict(X_val)
                    val_loss = loss(po1[T_val == 1], y_val[T_val == 1].to(DEVICE))
                    val_loss += loss(po0[T_val == 0], y_val[T_val == 0].to(DEVICE))
                    val_loss += self._ortho_reg()
                    if val_loss_best > val_loss:
                        val_loss_best = val_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience > self.patience and i > self.n_iter_min:
                        break
                    log.info(
                        f"Epoch: {i}, current {val_string} loss: {val_loss}, train_loss: {torch.mean(train_loss)}"
                    )
        return self


class Simulator:
    """
    Data generation process.

    Args:
        X: np.ndarray/pd.DataFrame
            Baseline covariates
        T: np.ndarray/pd.DataFrame
            Baseline treatments
        Y: np.ndarray/pd.DataFrame
            Baseline outcomes
        seed: int
            Random seed
        tau: float
            LeakyReLU parameter
        prognostic_mask: float
            Parameter to control the prognostic/predictive output
        lr: float
            Learning rate
        weight_decay: float
            Weight decay for the optimizer
        n_iter: int
            Number of training iterations
        batch_size: int
            Batch size used for training
        seed: int
            Random seed
        patience: int
            Number of iterations before early stopping
        n_iter_min: int
            Minimum number of iterations before early stopping
        Dropout: float
            probability of an element to be zeroed.
        n_units_hidden: int
            Size of the hidden layers
        n_layers: int
            Number of the hidden layers
        noise: bool
            If true, adds a small noise from N(0,1) to the outcomes

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
        T: np.ndarray,
        Y: np.ndarray,
        seed: int = 42,
        dropout: float = 0,
        batch_size: int = 50,
        n_units_hidden: int = 200,
        n_layers: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        penalty_orthogonal: float = 0.001,
        n_iter: int = 1000,
    ) -> None:

        enable_reproducible_results(seed=seed)
        self.seed = seed
        self.est = RawEstimator(
            X.shape[1],
            seed=seed,
            dropout=dropout,
            batch_size=batch_size,
            n_units_hidden=n_units_hidden,
            n_layers=n_layers,
            lr=lr,
            weight_decay=weight_decay,
            penalty_orthogonal=penalty_orthogonal,
            n_iter=n_iter,
            binary_y=(len(np.unique(Y)) == 2),
        )

        self.est.train(X, T, Y)

    def simulate_dataset(
        self,
        X: np.ndarray,
        tau: float = 1,
        prognostic_mask: float = 1,
        scale_factor: float = 1,
        treatment_assign: str = "random",
        noise: bool = True,
        err_std: float = 1,
        prop_scale: float = 1,
        addvar_scale: float = 0.1,
    ) -> Tuple:
        enable_reproducible_results(self.seed)
        self.scale_factor = scale_factor
        self.prognostic_mask = prognostic_mask
        self.treatment_assign = treatment_assign
        self.noise = noise
        self.err_std = err_std
        self.prop_scale = prop_scale
        self.addvar_scale = addvar_scale
        self.est.prog.set_tau(tau)
        self.est.pred0.set_tau(tau)
        self.est.pred1.set_tau(tau)
        self.top_idx = None
        _, _, prog, pred0, pred1 = self.est.predict(X)
        prog = prog.detach().cpu().numpy()
        pred0 = pred0.detach().cpu().numpy()
        pred1 = pred1.detach().cpu().numpy()

        top_idx = None
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

        if self.treatment_assign == "top_prog":
            po0 = self.scale_factor * (
                self.prognostic_mask * (prog + self.addvar_scale * X[:, top_idx])
                + pred0
            )
            po1 = self.scale_factor * (
                self.prognostic_mask * (prog + self.addvar_scale * X[:, top_idx])
                + pred1
            )
        elif self.treatment_assign == "top_pred":
            po0 = self.scale_factor * (self.prognostic_mask * prog + pred0)
            po1 = self.scale_factor * (
                self.prognostic_mask * prog + pred1 + self.addvar_scale * X[:, top_idx]
            )
        else:
            po0 = self.scale_factor * (self.prognostic_mask * prog + pred0)
            po1 = self.scale_factor * (self.prognostic_mask * prog + pred1)

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

        return X, W_synth, Y_synth, po0, po1, propensity, self.est, top_idx

    def augment_dataset(
        self,
        X: np.ndarray,
        W_synth: np.ndarray,
        Y_synth: np.ndarray,
        po0: np.ndarray,
        po1: np.ndarray,
        propensity: np.ndarray,
        size_total: int,
        alpha: float = 0.4,
    ) -> Tuple:
        n_mixup = size_total - len(X)
        if n_mixup < 0:
            raise (
                ValueError(
                    "The size of the augmented dataset should be larger than the size of the initial dataset."
                )
            )

        # Generate the mixup features:
        idx1 = np.random.randint(
            0, high=len(X), size=n_mixup
        )  # First sample feature indices
        idx2 = np.random.randint(
            0, high=len(X), size=n_mixup
        )  # Second sample feature indices
        weights_mixup = np.random.beta(
            alpha, alpha, size=(n_mixup, 1)
        )  # Weights sampled from beta distribution
        X_mixup = (
            weights_mixup * X[idx1] + (1 - weights_mixup) * X[idx2]
        )  # Weighted combination of the first and second sample features

        # Evaluate DGP labels for the mixed up features:
        _, _, prog_mixup, pred0_mixup, pred1_mixup = self.est.predict(X_mixup)
        prog_mixup = prog_mixup.detach().cpu().numpy()
        pred0_mixup = pred0_mixup.detach().cpu().numpy()
        pred1_mixup = pred1_mixup.detach().cpu().numpy()
        po0_mixup = self.scale_factor * (
            self.prognostic_mask * prog_mixup + pred0_mixup
        )
        po1_mixup = self.scale_factor * (
            self.prognostic_mask * prog_mixup + pred1_mixup
        )

        # Assign treatment (ADD CONFOUNDING WHEN AVAILABLE):
        W_synth_mixup = np.random.randint(2, size=n_mixup)
        propensity_mixup = 0.5 * np.ones(n_mixup)

        # Compute the final (possibly) noisy labels
        error_mixup = 0
        if self.noise:
            error_mixup = (
                torch.empty(n_mixup)
                .normal_(std=self.err_std)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
        Y_synth_mixup = (
            self.scale_factor
            * (
                self.prognostic_mask * prog_mixup
                + W_synth_mixup * pred1_mixup
                + (1 - W_synth_mixup) * pred0_mixup
            )
            + error_mixup
        )

        # Add the mixup dataset to the orignial dataset and return everything:
        X = np.concatenate((X, X_mixup))
        W_synth = np.concatenate((W_synth, W_synth_mixup))
        Y_synth = np.concatenate((Y_synth, Y_synth_mixup))
        po0 = np.concatenate((po0, po0_mixup))
        po1 = np.concatenate((po1, po1_mixup))
        propensity = np.concatenate((propensity, propensity_mixup))

        return X, W_synth, Y_synth, po0, po1, propensity

    def po0(self, X: torch.Tensor) -> torch.Tensor:

        if self.treatment_assign == "irrelevant_var":
            X[:, self.top_idx] = 0

        _, _, prog_factor, pred0_factor, _ = self.est.predict(X)
        if self.treatment_assign == "top_prog":
            po0 = self.scale_factor * (
                self.prognostic_mask
                * (prog_factor + self.addvar_scale * X[:, self.top_idx])
                + pred0_factor
            )
        else:
            po0 = self.scale_factor * (
                self.prognostic_mask * prog_factor + pred0_factor
            )
        return po0

    def po1(self, X: torch.Tensor) -> torch.Tensor:
        if self.treatment_assign == "irrelevant_var":
            X[:, self.top_idx] = 0

        _, _, prog_factor, _, pred1_factor = self.est.predict(X)
        if self.treatment_assign == "top_pred":
            po1 = self.scale_factor * (
                self.prognostic_mask * prog_factor
                + pred1_factor
                + self.addvar_scale * X[:, self.top_idx]
            )
        elif self.treatment_assign == "top_prog":
            po1 = self.scale_factor * (
                self.prognostic_mask
                * (prog_factor + self.addvar_scale * X[:, self.top_idx])
                + pred1_factor
            )
        else:
            po1 = self.scale_factor * (
                self.prognostic_mask * prog_factor + pred1_factor
            )
        return po1

    def te(self, X: torch.Tensor) -> torch.Tensor:
        if self.treatment_assign == "irrelevant_var":
            X[:, self.top_idx] = 0

        _, _, _, pred0_factor, pred1_factor = self.est.predict(X)
        if self.treatment_assign == "top_pred":
            te = self.scale_factor * (
                pred1_factor + self.addvar_scale * X[:, self.top_idx] - pred0_factor
            )
        else:
            te = self.scale_factor * (pred1_factor - pred0_factor)
        return te

    def prog(self, X: torch.Tensor) -> torch.Tensor:

        if self.treatment_assign == "irrelevant_var":
            X[:, self.top_idx] = 0

        _, _, prog_factor, _, _ = self.est.predict(X)
        if self.treatment_assign == "top_prog":
            return self.scale_factor * (
                prog_factor + self.addvar_scale * X[:, self.top_idx]
            )
        else:
            return self.scale_factor * prog_factor
