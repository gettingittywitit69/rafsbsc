"""Importable runtime helpers for ``notebooks/bsc_final.ipynb``."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import os

# Clamp BLAS/OpenMP threads before importing numpy/scipy so cell-level workers
# do not oversubscribe CPU threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from . import sharpe_mc

__all__ = [
    "ANALYTIC_METHOD",
    "GARCH_MLE_METHOD",
    "Config",
    "fit_model_to_data",
    "omega_garch_plugin",
    "run_cell",
    "run_partA",
    "se_iid_analytic",
    "sharpe_hat",
    "simulate_from_true_dgp",
    "simulate_garch11_t",
    "simulate_iid_normal",
    "stable_seed",
]


ANALYTIC_METHOD = "iid_normal_analytic"
GARCH_MLE_METHOD = "garch11_mle_analytic"
SUPPORTED_DGPS = ("iid_normal", "garch11_t")
SUPPORTED_METHODS = (ANALYTIC_METHOD, GARCH_MLE_METHOD)
MAXITER_COLD = 500
MAXITER_WARM = 200


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def sharpe_hat(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("x must be a 1d array with at least 2 observations")
    sd = float(np.std(arr, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    return float(np.mean(arr) / sd)


def _nanmean_or_nan(arr: np.ndarray) -> float:
    values = np.asarray(arr, dtype=float)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan


def _as_tuple(value: Any, caster) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(value)
    elif isinstance(value, np.ndarray):
        items = tuple(value.tolist())
    elif isinstance(value, str):
        items = (value,)
    else:
        try:
            items = tuple(value)
        except TypeError:
            items = (value,)
    return tuple(caster(item) for item in items)


def _vectorized_sharpe_hat(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("x must have shape (R, n) with n >= 2")
    mu_hat = np.mean(arr, axis=1)
    sd_hat = np.std(arr, axis=1, ddof=1)
    return np.divide(
        mu_hat,
        sd_hat,
        out=np.full(arr.shape[0], np.nan, dtype=float),
        where=np.isfinite(sd_hat) & (sd_hat > 0),
    )


def se_iid_analytic(S_hat: float | np.ndarray, n: int) -> float | np.ndarray:
    s_arr = np.asarray(S_hat, dtype=float)
    se = np.sqrt((1.0 + 0.5 * s_arr**2) / float(n))
    return float(se) if se.ndim == 0 else se


def _h2_from_nu(nu: float) -> float:
    nu_val = float(nu)
    if not np.isfinite(nu_val):
        return 3.0
    if nu_val <= 4.0:
        return np.nan
    return float(3.0 * (nu_val - 2.0) / (nu_val - 4.0))


def _valid_starting_values(
    starting_values: np.ndarray | None,
    expected_len: int,
) -> np.ndarray | None:
    if starting_values is None:
        return None
    arr = np.asarray(starting_values, dtype=float)
    if arr.ndim != 1 or arr.size != int(expected_len):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _failed_fit_result(
    *,
    model_name: str,
    distribution: str,
    nu: float = np.nan,
) -> dict[str, Any]:
    return {
        "model": str(model_name),
        "distribution": str(distribution),
        "mu": np.nan,
        "omega": np.nan,
        "alpha1": np.nan,
        "beta": np.nan,
        "gamma": np.nan,
        "nu": float(nu) if np.isfinite(nu) else np.nan,
        "h2": np.nan,
        "d": np.nan,
        "params_vec": None,
        "fit_failed": True,
        "convergence_flag": np.nan,
    }


def _finalize_garch11_fit(
    *,
    res: Any,
    params_vec: np.ndarray,
    model_name: str,
    distribution: str,
    fallback_nu: float = np.nan,
) -> dict[str, Any]:
    params = res.params
    alpha1 = float(params.get("alpha[1]", np.nan))
    beta = float(params.get("beta[1]", np.nan))
    gamma = alpha1 + beta if np.isfinite(alpha1) and np.isfinite(beta) else np.nan
    nu_fitted = float(params.get("nu", np.nan))
    nu = nu_fitted if np.isfinite(nu_fitted) else float(fallback_nu)
    h2 = 3.0 if distribution == "gaussian" else _h2_from_nu(nu)
    d = 1.0 - gamma**2 - (h2 - 1.0) * alpha1**2 if np.isfinite(gamma) and np.isfinite(h2) else np.nan

    convergence_flag = getattr(res, "convergence_flag", 0)
    try:
        optimizer_failed = int(convergence_flag) != 0
    except (TypeError, ValueError):
        optimizer_failed = bool(convergence_flag)

    # Treat optimizer and parameter pathologies conservatively so coverage is
    # unconditional on successful fits.
    fit_failed = optimizer_failed
    fit_failed = fit_failed or (not np.isfinite(alpha1)) or (not np.isfinite(beta))
    fit_failed = fit_failed or (alpha1 < 0.0) or (beta < 0.0)
    fit_failed = fit_failed or (not np.isfinite(gamma)) or (gamma >= 1.0)
    fit_failed = fit_failed or (not np.isfinite(h2)) or (not np.isfinite(d)) or (d <= 0.0)
    if distribution == "student_t":
        fit_failed = fit_failed or (not np.isfinite(nu)) or (nu <= 4.0)

    return {
        "model": str(model_name),
        "distribution": str(distribution),
        "mu": float(params.get("mu", np.nan)),
        "omega": float(params.get("omega", np.nan)),
        "alpha1": alpha1,
        "beta": beta,
        "gamma": gamma,
        "nu": nu if distribution == "student_t" else np.nan,
        "h2": h2,
        "d": d,
        "params_vec": np.asarray(params_vec, dtype=float),
        "fit_failed": bool(fit_failed),
        "convergence_flag": convergence_flag,
    }


def _fit_garch11_gaussian(
    x: np.ndarray,
    start: np.ndarray | None,
    maxiter: int,
) -> dict[str, Any]:
    try:
        _, res, params_vec = sharpe_mc.fit_candidate(
            np.asarray(x, dtype=float),
            "garch11_normal",
            starting_values=start,
            maxiter=int(maxiter),
        )
    except Exception:
        return _failed_fit_result(model_name="garch11_normal", distribution="gaussian")
    return _finalize_garch11_fit(
        res=res,
        params_vec=np.asarray(params_vec, dtype=float),
        model_name="garch11_normal",
        distribution="gaussian",
    )


def _fit_garch11_student_t(
    x: np.ndarray,
    start: np.ndarray | None,
    maxiter: int,
    fallback_nu: float,
) -> dict[str, Any]:
    try:
        _, res, params_vec = sharpe_mc.fit_candidate(
            np.asarray(x, dtype=float),
            "garch11_t",
            starting_values=start,
            maxiter=int(maxiter),
        )
    except Exception:
        return _failed_fit_result(
            model_name="garch11_t",
            distribution="student_t",
            nu=float(fallback_nu) if np.isfinite(fallback_nu) else np.nan,
        )
    return _finalize_garch11_fit(
        res=res,
        params_vec=np.asarray(params_vec, dtype=float),
        model_name="garch11_t",
        distribution="student_t",
        fallback_nu=float(fallback_nu),
    )


def fit_model_to_data(dgp: str, x: np.ndarray, cfg: "Config") -> dict[str, Any]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 10:
        raise ValueError("x must be a 1d array with at least 10 observations")
    if dgp == "iid_normal":
        fit_params = _fit_garch11_gaussian(arr, start=None, maxiter=MAXITER_COLD)
    elif dgp == "garch11_t":
        fit_params = _fit_garch11_student_t(
            arr,
            start=None,
            maxiter=MAXITER_COLD,
            fallback_nu=float(cfg.nu),
        )
    else:
        raise ValueError(f"Unknown model for fit: {dgp}")

    fit_params = dict(fit_params)
    fit_params["fit_source_dgp"] = str(dgp)
    return fit_params


def omega_garch_plugin(S_hat_vec: np.ndarray, fit_params: dict[str, Any]) -> np.ndarray:
    s_arr = np.asarray(S_hat_vec, dtype=float)
    distribution = str(fit_params.get("distribution", "student_t"))
    alpha1 = float(fit_params.get("alpha1", np.nan))
    beta = float(fit_params.get("beta", np.nan))
    gamma = float(fit_params.get("gamma", alpha1 + beta))
    nu = float(fit_params.get("nu", np.nan))
    h2_default = 3.0 if distribution == "gaussian" else _h2_from_nu(nu)
    h2 = float(fit_params.get("h2", h2_default))
    d = float(
        fit_params.get(
            "d",
            1.0 - gamma**2 - (h2 - 1.0) * alpha1**2 if np.isfinite(h2) else np.nan,
        )
    )

    invalid_fit = bool(fit_params.get("fit_failed", False))
    invalid_fit = invalid_fit or (not np.isfinite(alpha1)) or (not np.isfinite(beta))
    invalid_fit = invalid_fit or (alpha1 < 0.0) or (beta < 0.0)
    invalid_fit = invalid_fit or (not np.isfinite(gamma)) or (gamma >= 1.0)
    invalid_fit = invalid_fit or (distribution == "student_t" and np.isfinite(nu) and nu <= 4.0)
    invalid_fit = invalid_fit or (not np.isfinite(h2)) or (not np.isfinite(d)) or (d <= 0.0)
    if invalid_fit:
        omega = np.full_like(s_arr, np.nan, dtype=float)
        return float(omega) if omega.ndim == 0 else omega

    # Closed-form GARCH(1,1) plug-in asymptotic variance for the Sharpe Wald CI.
    factor = ((h2 - 1.0) * (1.0 + gamma) * (1.0 - beta) ** 2) / (d * (1.0 - gamma))
    omega = 1.0 + 0.25 * (s_arr**2) * factor
    invalid = (~np.isfinite(s_arr)) | (~np.isfinite(omega)) | (omega <= 0.0)
    omega = np.where(invalid, np.nan, np.asarray(omega, dtype=float))
    return float(omega) if omega.ndim == 0 else omega


def simulate_iid_normal(
    rng: np.random.Generator,
    n: int,
    S_true: float,
    sigma: float = 1.0,
    reps: int | None = None,
) -> np.ndarray:
    mu = float(S_true) * float(sigma)
    shape: int | tuple[int, int] = int(n) if reps is None else (int(reps), int(n))
    return mu + float(sigma) * rng.standard_normal(shape)


def simulate_garch11_t(
    rng: np.random.Generator,
    n: int,
    S_true: float,
    g_alpha: float,
    g_beta: float,
    nu: float,
    sigma_uncond: float = 1.0,
    burn: int = 500,
    reps: int | None = None,
) -> np.ndarray:
    if g_alpha < 0.0 or g_beta < 0.0 or g_alpha + g_beta >= 1.0:
        raise ValueError("Need g_alpha >= 0, g_beta >= 0, and g_alpha + g_beta < 1.")
    if nu <= 2.0:
        raise ValueError("Need nu > 2 to standardize t innovations to Var=1.")

    mu = float(S_true) * float(sigma_uncond)
    reps_int = 1 if reps is None else int(reps)
    total = int(n) + int(burn)
    omega = (1.0 - float(g_alpha) - float(g_beta)) * (float(sigma_uncond) ** 2)

    z = rng.standard_t(df=float(nu), size=(reps_int, total)) * np.sqrt((float(nu) - 2.0) / float(nu))
    x_out = np.empty((reps_int, int(n)), dtype=float)
    sigma2_prev = np.full(reps_int, float(sigma_uncond) ** 2, dtype=float)
    x_prev = mu + np.sqrt(sigma2_prev) * z[:, 0]
    if int(burn) == 0:
        x_out[:, 0] = x_prev

    for t in range(1, total):
        sigma2_prev = omega + float(g_alpha) * (x_prev - mu) ** 2 + float(g_beta) * sigma2_prev
        sigma2_prev = np.maximum(sigma2_prev, 1e-12)
        x_prev = mu + np.sqrt(sigma2_prev) * z[:, t]
        if t >= int(burn):
            x_out[:, t - int(burn)] = x_prev

    return x_out[0] if reps is None else x_out


def simulate_from_true_dgp(
    rng: np.random.Generator,
    dgp: str,
    n: int,
    S_true: float,
    cfg: "Config",
    reps: int | None = None,
) -> np.ndarray:
    if dgp == "iid_normal":
        return simulate_iid_normal(rng, n=n, S_true=S_true, sigma=1.0, reps=reps)
    if dgp == "garch11_t":
        return simulate_garch11_t(
            rng,
            n=n,
            S_true=S_true,
            g_alpha=cfg.g_alpha,
            g_beta=cfg.g_beta,
            nu=cfg.nu,
            sigma_uncond=1.0,
            burn=cfg.burn,
            reps=reps,
        )
    raise ValueError(f"Unknown dgp: {dgp}")


def _coverage_value(lo: np.ndarray, hi: np.ndarray, S_true: float) -> np.ndarray:
    covered = np.full(lo.shape, np.nan, dtype=float)
    valid = np.isfinite(lo) & np.isfinite(hi)
    covered[valid] = ((lo[valid] <= float(S_true)) & (float(S_true) <= hi[valid])).astype(float)
    return covered


def _reject_zero_value(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rejected = np.full(lo.shape, np.nan, dtype=float)
    valid = np.isfinite(lo) & np.isfinite(hi)
    rejected[valid] = ((lo[valid] > 0.0) | (hi[valid] < 0.0)).astype(float)
    return rejected


def _aggregate_method_row(
    *,
    dgp: str,
    n: int,
    S_true: float,
    method: str,
    outer_reps: int,
    se_arr: np.ndarray,
    omega_arr: np.ndarray,
    cov_arr: np.ndarray,
    rej_arr: np.ndarray,
    mc_sd: float,
    fit_fail_count: int | None = None,
) -> dict[str, Any]:
    finite_se = np.isfinite(se_arr)
    finite_omega = np.isfinite(omega_arr)
    mean_se = float(np.nanmean(se_arr)) if np.any(finite_se) else np.nan
    mean_omega = float(np.nanmean(omega_arr)) if np.any(finite_omega) else np.nan
    fail_count = int(np.count_nonzero(~finite_se)) if fit_fail_count is None else int(fit_fail_count)
    return {
        "dgp": str(dgp),
        "method": str(method),
        "n": int(n),
        "S_true": float(S_true),
        "outer_reps": int(outer_reps),
        "coverage_95": _nanmean_or_nan(cov_arr),
        "reject_rate_H0_S_eq_0": _nanmean_or_nan(rej_arr),
        "se_ratio_meanSE_over_mcSD": (mean_se / mc_sd) if np.isfinite(mean_se) and mc_sd > 0.0 else np.nan,
        "fit_fail_count": fail_count,
        "fit_fail_rate": float(fail_count / float(outer_reps)),
        "se_cell": mean_se,
        "omega_hat_cell": mean_omega,
    }


@dataclass(frozen=True)
class Config:
    seed: int = 0
    alpha: float = 0.05
    R: int = 60025
    # Keep the iid method at full R while allowing the expensive per-rep GARCH
    # MLE plug-in CI to run on a smaller replication budget.
    R_garch: int = 5000
    dgps: tuple[str, ...] = ("iid_normal", "garch11_t")
    methods: tuple[str, ...] = (ANALYTIC_METHOD, GARCH_MLE_METHOD)
    n_grid: tuple[int, ...] = (60, 120)
    S_grid: tuple[float, ...] = (
        -0.5,
        -0.4,
        -0.3,
        -0.2,
        -0.1,
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
    )
    g_alpha: float = 0.05
    g_beta: float = 0.90
    nu: float = 7.0
    burn: int = 500
    max_workers: int = max(1, (os.cpu_count() or 2) - 1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dgps", _as_tuple(self.dgps, str))
        object.__setattr__(self, "methods", _as_tuple(self.methods, str))
        object.__setattr__(self, "n_grid", _as_tuple(self.n_grid, int))
        object.__setattr__(self, "S_grid", _as_tuple(self.S_grid, float))
        if int(self.R) < 1:
            raise ValueError("R must be at least 1.")
        if int(self.R_garch) < 1 or int(self.R_garch) > int(self.R):
            raise ValueError("R_garch must satisfy 1 <= R_garch <= R.")
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must lie in (0, 1).")
        if int(self.burn) < 0:
            raise ValueError("burn must be non-negative.")
        if float(self.g_alpha) < 0.0 or float(self.g_beta) < 0.0 or float(self.g_alpha) + float(self.g_beta) >= 1.0:
            raise ValueError("Need g_alpha >= 0, g_beta >= 0, and g_alpha + g_beta < 1.")
        if float(self.nu) <= 2.0:
            raise ValueError("Need nu > 2 for the standardized t DGP.")
        if int(self.max_workers) < 1:
            raise ValueError("max_workers must be at least 1.")
        if not self.dgps:
            raise ValueError("dgps must not be empty.")
        if any(int(n) < 2 for n in self.n_grid):
            raise ValueError("All n values must be at least 2.")
        if not self.n_grid:
            raise ValueError("n_grid must not be empty.")
        if not self.S_grid:
            raise ValueError("S_grid must not be empty.")
        if not self.methods:
            raise ValueError("methods must not be empty.")
        unsupported_dgps = sorted(set(self.dgps) - set(SUPPORTED_DGPS))
        if unsupported_dgps:
            raise ValueError(f"Unsupported dgps: {unsupported_dgps}")
        unsupported_methods = sorted(set(self.methods) - set(SUPPORTED_METHODS))
        if unsupported_methods:
            raise ValueError(f"Unsupported methods: {unsupported_methods}")


def _coerce_config(cfg_like: Config | dict[str, Any]) -> Config:
    return cfg_like if isinstance(cfg_like, Config) else Config(**cfg_like)


def run_cell(
    dgp: str,
    n: int,
    S_true: float,
    cfg_like: Config | dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = _coerce_config(cfg_like)
    methods = set(cfg.methods)
    cell_rng = np.random.default_rng(stable_seed(cfg.seed, "cell", dgp, n, S_true))
    reps = int(cfg.R)
    garch_reps = int(cfg.R_garch)

    x = np.asarray(
        simulate_from_true_dgp(cell_rng, dgp=dgp, n=n, S_true=S_true, cfg=cfg, reps=reps),
        dtype=float,
    )
    if x.shape != (reps, int(n)):
        raise RuntimeError(f"Expected simulated shape {(reps, int(n))}, received {x.shape}.")

    s_hat = _vectorized_sharpe_hat(x)
    finite_s = s_hat[np.isfinite(s_hat)]
    mc_sd = float(np.std(finite_s, ddof=1)) if finite_s.size > 1 else np.nan
    s_hat_garch = s_hat[:garch_reps]
    finite_s_garch = s_hat_garch[np.isfinite(s_hat_garch)]
    mc_sd_garch = float(np.std(finite_s_garch, ddof=1)) if finite_s_garch.size > 1 else np.nan

    z = NormalDist().inv_cdf(1.0 - float(cfg.alpha) / 2.0)

    method_rows: list[dict[str, Any]] = []
    if ANALYTIC_METHOD in methods:
        omega_iid = 1.0 + 0.5 * s_hat**2
        se_iid = np.asarray(se_iid_analytic(s_hat, int(n)), dtype=float)
        lo_iid = s_hat - z * se_iid
        hi_iid = s_hat + z * se_iid
        cov_iid = _coverage_value(lo_iid, hi_iid, S_true)
        rej_iid = _reject_zero_value(lo_iid, hi_iid)
        method_rows.append(
            _aggregate_method_row(
                dgp=dgp,
                n=n,
                S_true=S_true,
                method=ANALYTIC_METHOD,
                outer_reps=reps,
                se_arr=se_iid,
                omega_arr=omega_iid,
                cov_arr=cov_iid,
                rej_arr=rej_iid,
                mc_sd=mc_sd,
            )
        )

    garch_fit_fail_count = 0
    garch_se = np.full(garch_reps, np.nan, dtype=float)
    garch_omega = np.full(garch_reps, np.nan, dtype=float)
    garch_cov = np.zeros(garch_reps, dtype=float)
    garch_rej = np.zeros(garch_reps, dtype=float)
    alpha1_hat = np.full(garch_reps, np.nan, dtype=float)
    beta_hat = np.full(garch_reps, np.nan, dtype=float)
    gamma_hat = np.full(garch_reps, np.nan, dtype=float)
    nu_hat = np.full(garch_reps, np.nan, dtype=float)
    h2_hat = np.full(garch_reps, np.nan, dtype=float)
    d_hat = np.full(garch_reps, np.nan, dtype=float)

    if GARCH_MLE_METHOD in methods:
        fit_one = _fit_garch11_gaussian if dgp == "iid_normal" else _fit_garch11_student_t
        expected_len = 4 if dgp == "iid_normal" else 5
        last_params_vec: np.ndarray | None = None

        # Both methods use the same simulated draws; the costly MLE path only
        # fits the first R_garch replications from that shared cell matrix.
        for rep_idx in range(garch_reps):
            start = _valid_starting_values(last_params_vec, expected_len)
            maxiter = MAXITER_WARM if start is not None else MAXITER_COLD
            if dgp == "iid_normal":
                fit = fit_one(x[rep_idx, :], start, maxiter)
            else:
                fit = fit_one(x[rep_idx, :], start, maxiter, float(cfg.nu))

            if bool(fit.get("fit_failed", True)):
                garch_fit_fail_count += 1
                continue

            omega_val = float(omega_garch_plugin(s_hat_garch[rep_idx], fit))
            se_val = float(np.sqrt(omega_val / float(n))) if np.isfinite(omega_val) and omega_val > 0.0 else np.nan
            if not np.isfinite(se_val):
                garch_fit_fail_count += 1
                continue

            lo = float(s_hat_garch[rep_idx] - z * se_val)
            hi = float(s_hat_garch[rep_idx] + z * se_val)
            if not np.isfinite(lo) or not np.isfinite(hi):
                garch_fit_fail_count += 1
                continue

            garch_cov[rep_idx] = float(lo <= float(S_true) <= hi)
            garch_rej[rep_idx] = float((lo > 0.0) or (hi < 0.0))
            garch_se[rep_idx] = se_val
            garch_omega[rep_idx] = omega_val
            alpha1_hat[rep_idx] = float(fit.get("alpha1", np.nan))
            beta_hat[rep_idx] = float(fit.get("beta", np.nan))
            gamma_hat[rep_idx] = float(fit.get("gamma", np.nan))
            nu_hat[rep_idx] = float(fit.get("nu", np.nan))
            h2_hat[rep_idx] = float(fit.get("h2", np.nan))
            d_hat[rep_idx] = float(fit.get("d", np.nan))

            next_start = _valid_starting_values(fit.get("params_vec"), expected_len)
            if next_start is not None:
                last_params_vec = next_start

        method_rows.append(
            _aggregate_method_row(
                dgp=dgp,
                n=n,
                S_true=S_true,
                method=GARCH_MLE_METHOD,
                outer_reps=garch_reps,
                se_arr=garch_se,
                omega_arr=garch_omega,
                cov_arr=garch_cov,
                rej_arr=garch_rej,
                mc_sd=mc_sd_garch,
                fit_fail_count=garch_fit_fail_count,
            )
        )

    garch_success = np.isfinite(garch_se)
    garch_ci_nan_count = int(np.count_nonzero(~garch_success)) if GARCH_MLE_METHOD in methods else 0
    diagnostics_row = {
        "dgp": str(dgp),
        "n": int(n),
        "S_true": float(S_true),
        "bias": _nanmean_or_nan(s_hat - float(S_true)),
        "rmse": float(np.sqrt(_nanmean_or_nan((s_hat - float(S_true)) ** 2))),
        "mc_sd_S_hat": mc_sd,
        "outer_reps": reps,
        "garch_outer_reps": garch_reps if GARCH_MLE_METHOD in methods else 0,
        "garch_fit_failed": bool(garch_fit_fail_count > 0) if GARCH_MLE_METHOD in methods else False,
        "garch_fit_fail_count": garch_fit_fail_count,
        "garch_fit_fail_rate": float(garch_fit_fail_count / garch_reps) if GARCH_MLE_METHOD in methods else np.nan,
        "garch_ci_nan_count": garch_ci_nan_count,
        "garch_ci_nan_rate": float(garch_ci_nan_count / garch_reps) if GARCH_MLE_METHOD in methods else np.nan,
        "garch_success_count": int(np.count_nonzero(garch_success)) if GARCH_MLE_METHOD in methods else 0,
        "garch_alpha1_hat": _nanmean_or_nan(alpha1_hat),
        "garch_beta_hat": _nanmean_or_nan(beta_hat),
        "garch_gamma_hat": _nanmean_or_nan(gamma_hat),
        "garch_nu_hat": _nanmean_or_nan(nu_hat),
        "garch_h2_hat": _nanmean_or_nan(h2_hat),
        "garch_d_hat": _nanmean_or_nan(d_hat),
    }
    return method_rows, diagnostics_row


def _run_cell_from_spec(
    spec: tuple[str, int, float, Config],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dgp, n, S_true, cfg = spec
    return run_cell(str(dgp), int(n), float(S_true), cfg)


def run_partA(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_specs = list(product(cfg.dgps, cfg.n_grid, cfg.S_grid))
    method_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []

    if int(cfg.max_workers) <= 1 or len(cell_specs) <= 1:
        cell_results = [
            _run_cell_from_spec((str(dgp), int(n), float(S_true), cfg))
            for dgp, n, S_true in cell_specs
        ]
    else:
        max_workers = min(int(cfg.max_workers), len(cell_specs))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            cell_results = list(
                executor.map(
                    _run_cell_from_spec,
                    (
                        (str(dgp), int(n), float(S_true), cfg)
                        for dgp, n, S_true in cell_specs
                    ),
                )
            )

    for rows_m, row_d in cell_results:
        method_rows.extend(rows_m)
        diagnostics_rows.append(row_d)

    df_methods = pd.DataFrame(method_rows).sort_values(
        ["dgp", "n", "S_true", "method"]
    ).reset_index(drop=True)
    df_diagnostics = pd.DataFrame(diagnostics_rows).sort_values(
        ["dgp", "n", "S_true"]
    ).reset_index(drop=True)
    return df_methods, df_diagnostics


def _env_scalar(name: str, default: Any, caster) -> Any:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else caster(raw.strip())


def _env_tuple(name: str, default: tuple[Any, ...], caster) -> tuple[Any, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(caster(part) for part in parts)


def config_from_env() -> Config:
    return Config(
        seed=_env_scalar("SEED", 0, int),
        alpha=_env_scalar("ALPHA", 0.05, float),
        R=60025,
        R_garch=_env_scalar("R_GARCH", 5000, int),
        dgps=_env_tuple("DGPS", ("iid_normal", "garch11_t"), str),
        methods=(ANALYTIC_METHOD, GARCH_MLE_METHOD),
        n_grid=_env_tuple("N_GRID", (60, 120), int),
        S_grid=_env_tuple(
            "S_GRID",
            (
                -0.5,
                -0.4,
                -0.3,
                -0.2,
                -0.1,
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
            ),
            float,
        ),
        g_alpha=_env_scalar("G_ALPHA", 0.05, float),
        g_beta=_env_scalar("G_BETA", 0.90, float),
        nu=_env_scalar("NU", 7.0, float),
        burn=_env_scalar("BURN", 500, int),
        max_workers=_env_scalar("MAX_WORKERS", max(1, (os.cpu_count() or 2) - 1), int),
    )


if __name__ == "__main__":
    cfg = config_from_env()
    results, diagnostics = run_partA(cfg)

    run_dir = Path(os.environ.get("EXPERIMENT_BSC_OUTPUT_DIR", "outputs/experiment_bsc")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results_partA_garch_mle_analytic.csv"
    diagnostics_path = run_dir / "results_partA_diagnostics.csv"

    results.to_csv(results_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    print(results)
    print(diagnostics)
    print()
    print(f"Wrote: {results_path}")
    print(f"Wrote: {diagnostics_path}")
