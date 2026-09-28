"""Per-objective Gaussian Process surrogate (sklearn).

Trained on a small set of analytically evaluated designs and used to predict
objective values for the population during NSGA-III generations. Periodic
retraining absorbs newly evaluated points (infill).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass
class SurrogateBank:
    """One GP per objective.

    ``log_obj_indices`` lists objective indices whose target should be modeled
    as ``log1p(max(0, y))`` to compress dynamic range (e.g. structural
    utilization which spans 0.03 to 5+).
    """

    n_obj: int
    rng: np.random.Generator
    log_obj_indices: tuple[int, ...] = (1,)  # structural utilization
    models: list[GaussianProcessRegressor] | None = None
    x_scaler: StandardScaler | None = None
    y_scaler: StandardScaler | None = None
    n_train: int = 0

    def _forward(self, y: np.ndarray) -> np.ndarray:
        out = y.copy()
        for j in self.log_obj_indices:
            out[:, j] = np.log1p(np.clip(y[:, j], 0.0, None))
        return out

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        out = y.copy()
        for j in self.log_obj_indices:
            out[:, j] = np.expm1(y[:, j])
        return out

    def _build_kernel(self, n_features: int):
        # ARD Matern (per-dimension length scales) lets the GP suppress
        # irrelevant input dimensions automatically. Critical for objectives
        # like structural utilization that depend on just 2 of 10 inputs.
        return (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(n_features),
                length_scale_bounds=(1e-2, 1e2),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.x_scaler = StandardScaler().fit(x_train)
        y_train_t = self._forward(y_train)
        self.y_scaler = StandardScaler().fit(y_train_t)
        x_s = self.x_scaler.transform(x_train)
        y_s = self.y_scaler.transform(y_train_t)

        n_features = x_train.shape[1]
        models: list[GaussianProcessRegressor] = []
        for j in range(self.n_obj):
            gp = GaussianProcessRegressor(
                kernel=self._build_kernel(n_features),
                normalize_y=False,
                alpha=1e-6,
                n_restarts_optimizer=4,
                random_state=int(self.rng.integers(0, 2**31 - 1)),
            )
            gp.fit(x_s, y_s[:, j])
            models.append(gp)
        self.models = models
        self.n_train = x_train.shape[0]

    def predict(self, x: np.ndarray, return_std: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        assert self.models is not None and self.x_scaler is not None and self.y_scaler is not None
        x_s = self.x_scaler.transform(x)
        means_s = np.empty((x.shape[0], self.n_obj))
        stds_s = np.empty((x.shape[0], self.n_obj))
        for j, gp in enumerate(self.models):
            mu, sd = gp.predict(x_s, return_std=True)
            means_s[:, j] = mu
            stds_s[:, j] = sd
        means_t = self.y_scaler.inverse_transform(means_s)
        means = self._inverse(means_t)
        if not return_std:
            return means
        # Approximate std in original units (multiply by per-dim scale).
        # For log-transformed dims this is approximate; we don't rely on it
        # for downstream selection.
        scales = self.y_scaler.scale_
        stds = stds_s * scales
        return means, stds

    def metrics(self, x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, list[float]]:
        """Per-objective RMSE, R^2, MAPE on a held-out set."""
        y_pred = self.predict(x_eval)
        rmse: list[float] = []
        r2: list[float] = []
        mape: list[float] = []
        for j in range(self.n_obj):
            err = y_pred[:, j] - y_eval[:, j]
            rmse.append(float(np.sqrt(np.mean(err ** 2))))
            ss_res = float(np.sum(err ** 2))
            ss_tot = float(np.sum((y_eval[:, j] - y_eval[:, j].mean()) ** 2))
            r2.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
            denom = np.where(np.abs(y_eval[:, j]) < 1e-9, 1e-9, np.abs(y_eval[:, j]))
            mape.append(float(np.mean(np.abs(err / denom)) * 100.0))
        return {"rmse": rmse, "r2": r2, "mape_pct": mape}

    @staticmethod
    def cv_metrics(
        x: np.ndarray,
        y: np.ndarray,
        n_obj: int,
        log_obj_indices: tuple[int, ...] = (1,),
        n_splits: int = 5,
        seed: int = 0,
    ) -> dict[str, list[float]]:
        """5-fold CV metrics on a frozen dataset. The most rigorous report."""
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        agg_r2: list[list[float]] = [[] for _ in range(n_obj)]
        agg_rmse: list[list[float]] = [[] for _ in range(n_obj)]
        agg_mape: list[list[float]] = [[] for _ in range(n_obj)]
        for fold_train, fold_eval in kf.split(x):
            bank = SurrogateBank(
                n_obj=n_obj,
                rng=np.random.default_rng(seed),
                log_obj_indices=log_obj_indices,
            )
            bank.fit(x[fold_train], y[fold_train])
            m = bank.metrics(x[fold_eval], y[fold_eval])
            for j in range(n_obj):
                agg_r2[j].append(m["r2"][j])
                agg_rmse[j].append(m["rmse"][j])
                agg_mape[j].append(m["mape_pct"][j])
        return {
            "r2": [float(np.mean(v)) for v in agg_r2],
            "rmse": [float(np.mean(v)) for v in agg_rmse],
            "mape_pct": [float(np.mean(v)) for v in agg_mape],
        }
