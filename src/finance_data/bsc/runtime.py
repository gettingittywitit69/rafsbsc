"""
Fast Part A runtime: cell-parallel, vectorized simulation, per-rep GARCH MLE plug-in CI.

Only supports:
  DGPs:    iid_normal, garch11_t
  Methods: iid_normal_analytic, garch11_mle_analytic, garch11_oracle_analytic
"""

from __future__ import annotations

import hashlib
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable

# clamp threads BEFORE numpy import (critical on multi-core nodes)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

from .garch_oracle import h2_from_innov, omega_garch_closed_form
from . import sharpe_mc

ANALYTIC_METHOD = "iid_normal_analytic"
GARCH_MLE_METHOD = "garch11_mle_analytic"
GARCH_ORACLE_METHOD = "garch11_oracle_analytic"

SUPPORTED_DGPS = ("iid_normal", "garch11_t")
SUPPORTED_METHODS = (ANALYTIC_METHOD, GARCH_MLE_METHOD, GARCH_ORACLE_METHOD)

MAXITER_COLD = 200
MAXITER_WARM = 80
MLE_TOL = 1e-6

EPS = 1e-12
_parallel_fallback_warned = False


def stable_seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16)


def _vectorized_sharpe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.mean(x, axis=1)
    sd = np.std(x, axis=1, ddof=1)
    out = np.full(x.shape[0], np.nan, dtype=float)
    ok = np.isfinite(sd) & (sd > 0)
    out[ok] = mu[ok] / sd[ok]
    return out


def se_iid_analytic(S_hat: np.ndarray, n: int) -> np.ndarray:
    return np.sqrt((1.0 + 0.5 * S_hat**2) / float(n))


def _h2_from_nu(nu: float) -> float:
    try:
        return float(h2_from_innov("t", nu))
    except Exception:
        return np.nan


def _safe_nu(nu_hat: float, fallback_nu: float) -> float:
    nu_hat = float(nu_hat)
    fallback_nu = float(fallback_nu)
    if np.isfinite(nu_hat) and nu_hat > 4.5:
        return min(nu_hat, 500.0)
    if np.isfinite(fallback_nu) and fallback_nu > 4.5:
        return min(fallback_nu, 500.0)
    return 8.0


def _squared_autocorr_lag1(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 3:
        return 0.0
    sq = (x - np.mean(x)) ** 2
    a = sq[:-1] - np.mean(sq[:-1])
    b = sq[1:] - np.mean(sq[1:])
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if not np.isfinite(den) or den <= EPS:
        return 0.0
    rho = float(np.dot(a, b) / den)
    if not np.isfinite(rho):
        return 0.0
    return float(np.clip(rho, 0.0, 0.98))


def _is_admissible_garch_plugin(alpha1: float, beta: float, h2: float) -> bool:
    alpha1 = float(alpha1)
    beta = float(beta)
    h2 = float(h2)
    gamma = alpha1 + beta
    d = 1.0 - gamma**2 - (h2 - 1.0) * alpha1**2
    return bool(
        np.isfinite(alpha1)
        and np.isfinite(beta)
        and np.isfinite(h2)
        and (alpha1 >= 0.0)
        and (beta >= 0.0)
        and (gamma < 1.0)
        and (h2 > 1.0)
        and (d > 0.0)
    )


def _project_garch_plugin_params(alpha1: float, beta: float, *, h2: float, gamma_fallback: float) -> tuple[float, float]:
    h2 = float(h2)
    gamma_fallback = float(gamma_fallback)

    gamma_raw = float(alpha1 + beta) if np.isfinite(alpha1) and np.isfinite(beta) else gamma_fallback
    gamma = float(np.clip(gamma_raw if np.isfinite(gamma_raw) else gamma_fallback, 0.0, 0.98))
    if gamma <= 0.0:
        return 0.0, 0.0

    alpha_default = float(min(0.05, 0.25 * gamma))
    alpha = float(alpha1) if np.isfinite(alpha1) and alpha1 >= 0.0 else alpha_default

    if not np.isfinite(h2) or h2 <= 1.0:
        h2 = 3.0
    alpha_cap = np.sqrt(max(1.0 - gamma**2, EPS) / max(h2 - 1.0, EPS))
    alpha_cap = float(min(gamma, 0.98 * alpha_cap))
    alpha = float(np.clip(alpha, 0.0, alpha_cap))
    beta_proj = float(max(gamma - alpha, 0.0))
    return alpha, beta_proj


def _moment_garch_plugin_params(x: np.ndarray, *, dist: str, fallback_nu: float) -> dict[str, float]:
    gamma = _squared_autocorr_lag1(x)
    if dist == "gaussian":
        h2 = 3.0
    else:
        h2 = _h2_from_nu(_safe_nu(np.nan, fallback_nu))
        if not np.isfinite(h2):
            h2 = 3.0
    alpha1, beta = _project_garch_plugin_params(
        alpha1=min(0.05, 0.25 * gamma),
        beta=max(gamma - min(0.05, 0.25 * gamma), 0.0),
        h2=h2,
        gamma_fallback=gamma,
    )
    return {"alpha1": float(alpha1), "beta": float(beta), "h2": float(h2)}


def _initial_garch_start(dgp: str, S_true: float, cfg: "Config", *, dist: str) -> dict[str, float] | None:
    if dist == "gaussian":
        return {
            "mu": float(S_true),
            "omega": 1.0,
            "alpha[1]": 0.0,
            "beta[1]": 0.0,
        }
    if dgp == "garch11_t":
        return {
            "mu": float(S_true),
            "omega": float(max(1.0 - float(cfg.g_alpha) - float(cfg.g_beta), EPS)),
            "alpha[1]": float(cfg.g_alpha),
            "beta[1]": float(cfg.g_beta),
            "nu": float(cfg.nu),
        }
    return None


def omega_garch_plugin(S_hat: float, *, alpha1: float, beta: float, h2: float) -> float:
    omega = omega_garch_closed_form(float(S_hat), float(alpha1), float(beta), float(h2))
    return float(omega) if np.isfinite(omega) else np.nan


def simulate_iid_normal(rng: np.random.Generator, n: int, S_true: float, reps: int) -> np.ndarray:
    mu = float(S_true)
    return mu + rng.standard_normal((int(reps), int(n)))


def simulate_garch11_t(
    rng: np.random.Generator,
    n: int,
    S_true: float,
    reps: int,
    *,
    g_alpha: float,
    g_beta: float,
    nu: float | None,
    burn: int,
    dist: str = "t",
) -> np.ndarray:
    if g_alpha < 0.0 or g_beta < 0.0 or g_alpha + g_beta >= 1.0:
        raise ValueError("Need g_alpha,g_beta>=0 and g_alpha+g_beta<1.")

    n = int(n)
    reps = int(reps)
    burn = int(burn)
    total = n + burn
    mu = float(S_true)

    omega = (1.0 - float(g_alpha) - float(g_beta))  # since uncond sigma^2 = 1
    dist_name = str(dist).strip().lower()
    if dist_name in {"normal", "gaussian"}:
        z = rng.standard_normal((reps, total))
    elif dist_name in {"t", "student_t", "student-t"}:
        if nu is None:
            raise ValueError("Need nu>2 for t innovations.")
        nu_f = float(nu)
        if nu_f <= 2.0:
            raise ValueError("Need nu>2 to standardize Var=1.")
        z = rng.standard_t(df=nu_f, size=(reps, total)) * np.sqrt((nu_f - 2.0) / nu_f)
    else:
        raise ValueError(f"Unsupported innovation distribution: {dist}")

    out = np.empty((reps, n), dtype=float)
    s2 = np.ones(reps, dtype=float)
    x_prev = mu + np.sqrt(s2) * z[:, 0]

    for t in range(1, total):
        s2 = omega + float(g_alpha) * (x_prev - mu) ** 2 + float(g_beta) * s2
        s2 = np.maximum(s2, EPS)
        x_prev = mu + np.sqrt(s2) * z[:, t]
        if t >= burn:
            out[:, t - burn] = x_prev

    if burn == 0:
        out[:, 0] = mu + z[:, 0]
    return out


def simulate_garch11_t_stats(
    rng: np.random.Generator,
    n: int,
    S_true: float,
    reps: int,
    *,
    g_alpha: float,
    g_beta: float,
    nu: float | None,
    burn: int,
    store_reps: int,
    sim_dtype: str = "float64",
    dist: str = "t",
) -> tuple[np.ndarray, np.ndarray | None]:
    if g_alpha < 0.0 or g_beta < 0.0 or g_alpha + g_beta >= 1.0:
        raise ValueError("Need g_alpha,g_beta>=0 and g_alpha+g_beta<1.")

    n = int(n)
    reps = int(reps)
    burn = int(burn)
    store_reps = int(max(0, min(store_reps, reps)))
    total = n + burn

    alpha = float(g_alpha)
    beta = float(g_beta)
    omega = float(1.0 - alpha - beta)
    mu = float(S_true)
    dist_name = str(dist).strip().lower()
    nu_f = float(nu) if nu is not None else np.nan
    scale = float(np.sqrt((nu_f - 2.0) / nu_f)) if np.isfinite(nu_f) and nu_f > 2.0 else np.nan
    eps = float(EPS)

    use_float32 = str(sim_dtype).lower() == "float32"
    x_dtype = np.float32 if use_float32 else np.float64

    s2 = np.ones(reps, dtype=np.float64)
    x_prev = np.empty(reps, dtype=x_dtype)
    x_t = np.empty(reps, dtype=np.float64)
    resid_sq = np.empty(reps, dtype=np.float64)
    sqrt_s2 = np.empty(reps, dtype=np.float64)
    sum_x = np.zeros(reps, dtype=np.float64)
    sum_x2 = np.zeros(reps, dtype=np.float64)

    x_fit = np.empty((store_reps, n), dtype=np.float64) if store_reps > 0 else None

    for t in range(total):
        if dist_name in {"normal", "gaussian"}:
            z_t = rng.standard_normal(size=reps)
        elif dist_name in {"t", "student_t", "student-t"}:
            if not np.isfinite(nu_f) or nu_f <= 2.0:
                raise ValueError("Need nu>2 to standardize t innovations to Var=1.")
            z_t = rng.standard_t(df=nu_f, size=reps)
            if use_float32:
                z_t = z_t.astype(np.float32, copy=False)
            z_t *= scale
        else:
            raise ValueError(f"Unsupported innovation distribution: {dist}")

        if use_float32 and dist_name in {"normal", "gaussian"}:
            z_t = z_t.astype(np.float32, copy=False)

        if t > 0:
            np.subtract(x_prev, mu, out=resid_sq)
            np.square(resid_sq, out=resid_sq)
            np.multiply(s2, beta, out=s2)
            s2 += omega
            s2 += alpha * resid_sq
            np.maximum(s2, eps, out=s2)

        np.sqrt(s2, out=sqrt_s2)
        np.multiply(sqrt_s2, z_t, out=x_t)
        x_t += mu

        if t >= burn:
            sample_idx = t - burn
            sum_x += x_t
            np.square(x_t, out=resid_sq)
            sum_x2 += resid_sq
            if x_fit is not None:
                x_fit[:, sample_idx] = x_t[:store_reps]

        x_prev[:] = x_t

    if n <= 1:
        s_hat = np.full(reps, np.nan, dtype=np.float64)
    else:
        n_f = float(n)
        mean = sum_x / n_f
        var = (sum_x2 - (sum_x * sum_x) / n_f) / float(n - 1)
        np.maximum(var, 0.0, out=var)
        sd = np.sqrt(var)
        s_hat = np.full(reps, np.nan, dtype=np.float64)
        ok = np.isfinite(sd) & (sd > 0.0)
        s_hat[ok] = mean[ok] / sd[ok]

    return s_hat, x_fit


def _fit_garch11(
    x: np.ndarray,
    *,
    dist: str,
    start: np.ndarray | dict[str, float] | None,
    maxiter: int,
    fallback_nu: float,
    tol: float = MLE_TOL,
) -> dict[str, Any]:
    name = "garch11_normal" if dist == "gaussian" else "garch11_t"
    gamma_hint = _squared_autocorr_lag1(x)
    fit_attempts = 2 if start is not None else 1
    fit_time_ms = np.nan
    fit_converged = False

    try:
        _, res, params_vec = sharpe_mc.fit_candidate(
            x,
            name,
            starting_values=start,
            maxiter=int(maxiter),
            tol=float(tol),
        )
        meta = getattr(res, "_fit_meta", None)
        if isinstance(meta, dict):
            fit_attempts = int(meta.get("fit_attempts", 1))
            fit_time_ms = float(meta.get("fit_time_ms", np.nan))
            fit_converged = bool(meta.get("fit_converged", sharpe_mc._fit_succeeded(res)))
        else:
            fit_converged = bool(sharpe_mc._fit_succeeded(res))
        if not fit_converged:
            raise RuntimeError("ARCH fit did not converge")
    except Exception:
        fallback = _moment_garch_plugin_params(x, dist=dist, fallback_nu=fallback_nu)
        return {
            "ok": True,
            "params_vec": None,
            "regularized": True,
            "fit_converged": False,
            "fit_attempts": int(fit_attempts),
            "fit_time_ms": float(fit_time_ms),
            **fallback,
        }

    p = res.params
    alpha1 = float(p.get("alpha[1]", np.nan))
    beta = float(p.get("beta[1]", np.nan))
    nu_hat = float(p.get("nu", np.nan))

    if dist == "gaussian":
        h2 = 3.0
        raw_ok = _is_admissible_garch_plugin(alpha1, beta, h2)
        h2_changed = False
    else:
        h2_raw = _h2_from_nu(nu_hat)
        h2 = _h2_from_nu(_safe_nu(nu_hat, fallback_nu))
        raw_ok = _is_admissible_garch_plugin(alpha1, beta, h2_raw)
        h2_changed = not np.isfinite(h2_raw) or abs(h2 - h2_raw) > 1e-10

    alpha1_proj, beta_proj = _project_garch_plugin_params(alpha1, beta, h2=h2, gamma_fallback=gamma_hint)
    if _is_admissible_garch_plugin(alpha1_proj, beta_proj, h2):
        return {
            "ok": True,
            "alpha1": float(alpha1_proj),
            "beta": float(beta_proj),
            "h2": float(h2),
            "params_vec": np.asarray(params_vec, float) if raw_ok else None,
            "fit_converged": bool(fit_converged),
            "fit_attempts": int(fit_attempts),
            "fit_time_ms": float(fit_time_ms),
            "regularized": bool(
                not raw_ok
                or h2_changed
                or abs(alpha1_proj - alpha1) > 1e-10
                or abs(beta_proj - beta) > 1e-10
            ),
        }

    fallback = _moment_garch_plugin_params(x, dist=dist, fallback_nu=fallback_nu)
    return {
        "ok": True,
        "params_vec": None,
        "regularized": True,
        "fit_converged": bool(fit_converged),
        "fit_attempts": int(fit_attempts),
        "fit_time_ms": float(fit_time_ms),
        **fallback,
    }


def _nanmean_or_nan(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.any(np.isfinite(x)) else float("nan")


def _coverage_with_mc_stats(cov: np.ndarray, outer_reps: int) -> dict[str, float]:
    cov = np.asarray(cov, dtype=float)
    finite = np.isfinite(cov)
    if not np.any(finite):
        return {"coverage_95": np.nan, "mc_se": np.nan, "mc_lo": np.nan, "mc_hi": np.nan}

    p_hat = float(np.mean(cov[finite]))
    reps = int(outer_reps)
    if reps < 1:
        return {"coverage_95": p_hat, "mc_se": np.nan, "mc_lo": np.nan, "mc_hi": np.nan}

    var = max(p_hat * (1.0 - p_hat), 0.0)
    mc_se = float(np.sqrt(var / float(reps)))
    return {
        "coverage_95": p_hat,
        "mc_se": mc_se,
        "mc_lo": float(p_hat - 1.96 * mc_se),
        "mc_hi": float(p_hat + 1.96 * mc_se),
    }


def normalize_ci_levels(ci_levels: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    levels = tuple(float(level) for level in ci_levels)
    if len(levels) == 0:
        raise ValueError("ci_levels must contain at least one confidence level.")
    if any((not np.isfinite(level)) or (level <= 0.0) or (level >= 1.0) for level in levels):
        raise ValueError("Each ci_levels entry must be finite and strictly between 0 and 1.")
    return levels


def _build_ci_sweep_rows(
    *,
    dgp: str,
    method: str,
    n: int,
    S_true: float,
    ci_levels: tuple[float, ...],
    s_hat: np.ndarray,
    se: np.ndarray,
    outer_reps: int,
    fit_fail_count: int,
    fit_fail_rate: float,
) -> list[dict[str, Any]]:
    s_arr = np.asarray(s_hat, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    rows: list[dict[str, Any]] = []

    for ci_level in ci_levels:
        z = NormalDist().inv_cdf(0.5 + 0.5 * float(ci_level))
        half_width = z * se_arr
        lo = s_arr - half_width
        hi = s_arr + half_width
        cov = np.full(s_arr.shape[0], np.nan, dtype=float)
        ok = np.isfinite(lo) & np.isfinite(hi)
        cov[ok] = ((lo[ok] <= float(S_true)) & (float(S_true) <= hi[ok])).astype(float)
        cov_stats = _coverage_with_mc_stats(cov, outer_reps)
        rows.append(
            {
                "dgp": dgp,
                "method": method,
                "n": int(n),
                "S_true": float(S_true),
                "ci_level": float(ci_level),
                "outer_reps": int(outer_reps),
                "coverage": cov_stats["coverage_95"],
                "avg_ci_length": _nanmean_or_nan(2.0 * np.abs(half_width)),
                "mc_se": cov_stats["mc_se"],
                "mc_lo": cov_stats["mc_lo"],
                "mc_hi": cov_stats["mc_hi"],
                "fit_fail_count": int(fit_fail_count),
                "fit_fail_rate": float(fit_fail_rate),
            }
        )
    return rows


@dataclass(frozen=True)
class Config:
    seed: int = 0
    alpha: float = 0.05
    R: int = 60025
    R_garch: int = 1000
    dgps: tuple[str, ...] = ("iid_normal", "garch11_t")
    methods: tuple[str, ...] = (ANALYTIC_METHOD, GARCH_MLE_METHOD, GARCH_ORACLE_METHOD)
    n_grid: tuple[int, ...] = (30, 60)
    S_grid: tuple[float, ...] = (-0.5, 0.0, 0.5)
    g_alpha: float = 0.05
    g_beta: float = 0.90
    garch_dist: str = "t"
    nu: float = 7.0
    burn: int = 500
    mle_maxiter_warm: int = MAXITER_WARM
    mle_maxiter_cold: int = MAXITER_COLD
    mle_tol: float = MLE_TOL
    max_workers: int = max(1, (os.cpu_count() or 2) - 1)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dgps", tuple(self.dgps))
        object.__setattr__(self, "methods", tuple(self.methods))
        object.__setattr__(self, "n_grid", tuple(int(n) for n in self.n_grid))
        object.__setattr__(self, "S_grid", tuple(float(s) for s in self.S_grid))
        object.__setattr__(self, "garch_dist", str(self.garch_dist).strip().lower())

        if self.R < 1:
            raise ValueError("R must be >= 1")
        if self.R_garch < 1:
            raise ValueError("R_garch must be >= 1")
        if self.R_garch > self.R:
            object.__setattr__(self, "R_garch", int(self.R))
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError("alpha must be in (0,1)")
        if self.burn < 0:
            raise ValueError("burn must be >= 0")
        if int(self.mle_maxiter_warm) < 1:
            raise ValueError("mle_maxiter_warm must be >= 1")
        if int(self.mle_maxiter_cold) < 1:
            raise ValueError("mle_maxiter_cold must be >= 1")
        if not np.isfinite(float(self.mle_tol)) or float(self.mle_tol) <= 0.0:
            raise ValueError("mle_tol must be finite and > 0")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.garch_dist not in ("t", "normal", "gaussian", "student_t", "student-t"):
            raise ValueError("garch_dist must be one of {'t','normal'} (synonyms allowed).")
        if any(d not in SUPPORTED_DGPS for d in self.dgps):
            raise ValueError(f"Unsupported dgps: {sorted(set(self.dgps) - set(SUPPORTED_DGPS))}")
        if any(m not in SUPPORTED_METHODS for m in self.methods):
            raise ValueError(f"Unsupported methods: {sorted(set(self.methods) - set(SUPPORTED_METHODS))}")


def _coerce_config(cfg: Config | dict[str, Any]) -> Config:
    if isinstance(cfg, Config):
        if int(cfg.R_garch) <= int(cfg.R):
            return cfg
        payload = cfg.__dict__.copy()
        payload["R_garch"] = int(cfg.R)
        return Config(**payload)

    payload = dict(cfg)
    if "R" in payload and "R_garch" in payload:
        payload["R_garch"] = min(int(payload["R_garch"]), int(payload["R"]))
    return Config(**payload)


def _run_cell_impl(
    dgp: str,
    n: int,
    S_true: float,
    cfg: Config | dict[str, Any],
    *,
    ci_levels: tuple[float, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cfg = _coerce_config(cfg)
    ci_levels = tuple(ci_levels) if ci_levels is not None else tuple()
    n = int(n)
    S_true = float(S_true)
    R = int(cfg.R)
    Rg = min(int(cfg.R_garch), R)
    garch_dist = "normal" if str(cfg.garch_dist).strip().lower() in {"normal", "gaussian"} else "t"

    rng = np.random.default_rng(stable_seed(cfg.seed, "cell", dgp, n, S_true))
    x_for_fit: np.ndarray | None = None
    if dgp == "iid_normal":
        x = simulate_iid_normal(rng, n=n, S_true=S_true, reps=R)
        s_hat = _vectorized_sharpe(x)
        x_for_fit = x
    elif dgp == "garch11_t":
        store_reps = Rg if GARCH_MLE_METHOD in cfg.methods else 0
        s_hat, x_for_fit = simulate_garch11_t_stats(
            rng,
            n=n,
            S_true=S_true,
            reps=R,
            g_alpha=cfg.g_alpha,
            g_beta=cfg.g_beta,
            nu=(cfg.nu if garch_dist == "t" else None),
            burn=cfg.burn,
            store_reps=store_reps,
            dist=garch_dist,
        )
    else:
        raise ValueError(f"Unsupported dgp: {dgp}")

    z = NormalDist().inv_cdf(1.0 - float(cfg.alpha) / 2.0)

    rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []

    # iid analytic (vectorized, cheap)
    if ANALYTIC_METHOD in cfg.methods:
        se = se_iid_analytic(s_hat, n)
        lo = s_hat - z * se
        hi = s_hat + z * se
        cov = ((lo <= S_true) & (S_true <= hi)).astype(float)
        rej = ((lo > 0.0) | (hi < 0.0)).astype(float)
        cov_stats = _coverage_with_mc_stats(cov, R)
        mc_sd = float(np.std(s_hat[np.isfinite(s_hat)], ddof=1)) if np.count_nonzero(np.isfinite(s_hat)) > 1 else np.nan
        row = {
            "dgp": dgp,
            "method": ANALYTIC_METHOD,
            "n": n,
            "S_true": S_true,
            "outer_reps": int(R),
            "coverage_95": cov_stats["coverage_95"],
            "mc_se": cov_stats["mc_se"],
            "mc_lo": cov_stats["mc_lo"],
            "mc_hi": cov_stats["mc_hi"],
            "reject_rate_H0_S_eq_0": _nanmean_or_nan(rej),
            "se_cell": _nanmean_or_nan(se),
            "omega_hat_cell": _nanmean_or_nan(1.0 + 0.5 * s_hat**2),
            "se_ratio_meanSE_over_mcSD": (_nanmean_or_nan(se) / mc_sd) if np.isfinite(mc_sd) and mc_sd > 0 else np.nan,
            "fit_fail_count": int(np.count_nonzero(~np.isfinite(se))),
            "fit_fail_rate": float(np.count_nonzero(~np.isfinite(se)) / float(R)),
        }
        rows.append(row)
        if ci_levels:
            ci_rows.extend(
                _build_ci_sweep_rows(
                    dgp=dgp,
                    method=ANALYTIC_METHOD,
                    n=n,
                    S_true=S_true,
                    ci_levels=ci_levels,
                    s_hat=s_hat,
                    se=se,
                    outer_reps=R,
                    fit_fail_count=int(row["fit_fail_count"]),
                    fit_fail_rate=float(row["fit_fail_rate"]),
                )
            )

    # garch plug-in MLE CI (loop, expensive, warm-start allowed within cell)
    garch_fail = 0
    garch_regularized = 0
    garch_fit_converged_rate = np.nan
    garch_fit_attempts_mean = np.nan
    garch_fit_time_ms_mean = np.nan
    if GARCH_MLE_METHOD in cfg.methods:
        s_g = s_hat[:Rg]

        se_g = np.full(Rg, np.nan, dtype=float)
        omega_g = np.full(Rg, np.nan, dtype=float)
        cov_g = np.full(Rg, np.nan, dtype=float)
        rej_g = np.full(Rg, np.nan, dtype=float)
        fit_converged_g = np.full(Rg, np.nan, dtype=float)
        fit_attempts_g = np.full(Rg, np.nan, dtype=float)
        fit_time_ms_g = np.full(Rg, np.nan, dtype=float)

        last_params: np.ndarray | None = None
        dist = "gaussian" if (dgp == "iid_normal" or garch_dist == "normal") else "student_t"
        first_start = _initial_garch_start(dgp, S_true, cfg, dist=dist)
        if x_for_fit is None:
            raise RuntimeError("GARCH MLE requested but no stored fit series are available.")

        for i in range(Rg):
            start = (
                last_params
                if (last_params is not None and last_params.ndim == 1 and np.all(np.isfinite(last_params)))
                else (first_start if i == 0 else None)
            )
            maxiter = int(cfg.mle_maxiter_warm) if start is not None else int(cfg.mle_maxiter_cold)
            fit = _fit_garch11(
                x_for_fit[i, :],
                dist=dist,
                start=start,
                maxiter=maxiter,
                fallback_nu=cfg.nu,
                tol=cfg.mle_tol,
            )
            fit_converged_g[i] = float(bool(fit.get("fit_converged", False)))
            fit_attempts_g[i] = float(fit.get("fit_attempts", np.nan))
            fit_time_ms_g[i] = float(fit.get("fit_time_ms", np.nan))
            if not fit.get("ok", False):
                garch_fail += 1
                continue
            if fit.get("regularized", False):
                garch_regularized += 1

            om = omega_garch_plugin(float(s_g[i]), alpha1=float(fit["alpha1"]), beta=float(fit["beta"]), h2=float(fit["h2"]))
            se = float(np.sqrt(om / float(n))) if np.isfinite(om) else np.nan
            if not np.isfinite(se):
                garch_fail += 1
                continue

            lo = float(s_g[i] - z * se)
            hi = float(s_g[i] + z * se)

            cov_g[i] = float(lo <= S_true <= hi)
            rej_g[i] = float((lo > 0.0) or (hi < 0.0))
            se_g[i] = se
            omega_g[i] = om

            candidate = fit.get("params_vec")
            if isinstance(candidate, np.ndarray) and candidate.ndim == 1 and np.all(np.isfinite(candidate)):
                last_params = candidate

        garch_fit_converged_rate = _nanmean_or_nan(fit_converged_g)
        garch_fit_attempts_mean = _nanmean_or_nan(fit_attempts_g)
        garch_fit_time_ms_mean = _nanmean_or_nan(fit_time_ms_g)
        cov_stats_g = _coverage_with_mc_stats(cov_g, Rg)

        mc_sd_g = float(np.std(s_g[np.isfinite(s_g)], ddof=1)) if np.count_nonzero(np.isfinite(s_g)) > 1 else np.nan
        row = {
            "dgp": dgp,
            "method": GARCH_MLE_METHOD,
            "n": n,
            "S_true": S_true,
            "outer_reps": int(Rg),
            "coverage_95": cov_stats_g["coverage_95"],
            "mc_se": cov_stats_g["mc_se"],
            "mc_lo": cov_stats_g["mc_lo"],
            "mc_hi": cov_stats_g["mc_hi"],
            "reject_rate_H0_S_eq_0": _nanmean_or_nan(rej_g),
            "se_cell": _nanmean_or_nan(se_g),
            "omega_hat_cell": _nanmean_or_nan(omega_g),
            "se_ratio_meanSE_over_mcSD": (_nanmean_or_nan(se_g) / mc_sd_g) if np.isfinite(mc_sd_g) and mc_sd_g > 0 else np.nan,
            "fit_fail_count": int(garch_fail),
            "fit_fail_rate": float(garch_fail / float(Rg)),
            "regularized_count": int(garch_regularized),
            "regularized_rate": float(garch_regularized / float(Rg)),
            "fit_converged_rate": garch_fit_converged_rate,
            "fit_attempts_mean": garch_fit_attempts_mean,
            "fit_time_ms_mean": garch_fit_time_ms_mean,
        }
        rows.append(row)
        if ci_levels:
            ci_rows.extend(
                _build_ci_sweep_rows(
                    dgp=dgp,
                    method=GARCH_MLE_METHOD,
                    n=n,
                    S_true=S_true,
                    ci_levels=ci_levels,
                    s_hat=s_g,
                    se=se_g,
                    outer_reps=Rg,
                    fit_fail_count=int(row["fit_fail_count"]),
                    fit_fail_rate=float(row["fit_fail_rate"]),
                )
            )

    if GARCH_ORACLE_METHOD in cfg.methods and dgp == "garch11_t":
        try:
            h2 = h2_from_innov(garch_dist, cfg.nu if garch_dist == "t" else None)
        except Exception:
            h2 = np.nan

        omega_raw = omega_garch_closed_form(s_hat, float(cfg.g_alpha), float(cfg.g_beta), float(h2))
        omega_o = np.asarray(omega_raw, dtype=float)
        if omega_o.ndim == 0:
            omega_o = np.full(R, float(omega_o), dtype=float)
        bad = ~np.isfinite(omega_o) | (omega_o <= 0.0)
        omega_o[bad] = np.nan

        se_o = np.full(R, np.nan, dtype=float)
        ok = np.isfinite(omega_o)
        se_o[ok] = np.sqrt(omega_o[ok] / float(n))

        lo = s_hat - z * se_o
        hi = s_hat + z * se_o
        cov_o = np.full(R, np.nan, dtype=float)
        rej_o = np.full(R, np.nan, dtype=float)
        cov_o[ok] = ((lo[ok] <= S_true) & (S_true <= hi[ok])).astype(float)
        rej_o[ok] = ((lo[ok] > 0.0) | (hi[ok] < 0.0)).astype(float)
        cov_stats_o = _coverage_with_mc_stats(cov_o, R)
        mc_sd_o = float(np.std(s_hat[np.isfinite(s_hat)], ddof=1)) if np.count_nonzero(np.isfinite(s_hat)) > 1 else np.nan
        fail_count_o = int(np.count_nonzero(~np.isfinite(se_o)))

        row = {
            "dgp": dgp,
            "method": GARCH_ORACLE_METHOD,
            "n": n,
            "S_true": S_true,
            "outer_reps": int(R),
            "coverage_95": cov_stats_o["coverage_95"],
            "mc_se": cov_stats_o["mc_se"],
            "mc_lo": cov_stats_o["mc_lo"],
            "mc_hi": cov_stats_o["mc_hi"],
            "reject_rate_H0_S_eq_0": _nanmean_or_nan(rej_o),
            "se_cell": _nanmean_or_nan(se_o),
            "omega_hat_cell": _nanmean_or_nan(omega_o),
            "se_ratio_meanSE_over_mcSD": (_nanmean_or_nan(se_o) / mc_sd_o) if np.isfinite(mc_sd_o) and mc_sd_o > 0 else np.nan,
            "fit_fail_count": int(fail_count_o),
            "fit_fail_rate": float(fail_count_o / float(R)),
            "regularized_count": 0,
            "regularized_rate": 0.0,
            "fit_converged_rate": np.nan,
            "fit_attempts_mean": np.nan,
            "fit_time_ms_mean": np.nan,
        }
        rows.append(row)
        if ci_levels:
            ci_rows.extend(
                _build_ci_sweep_rows(
                    dgp=dgp,
                    method=GARCH_ORACLE_METHOD,
                    n=n,
                    S_true=S_true,
                    ci_levels=ci_levels,
                    s_hat=s_hat,
                    se=se_o,
                    outer_reps=R,
                    fit_fail_count=int(row["fit_fail_count"]),
                    fit_fail_rate=float(row["fit_fail_rate"]),
                )
            )

    diag = {
        "dgp": dgp,
        "n": n,
        "S_true": S_true,
        "bias": _nanmean_or_nan(s_hat - S_true),
        "rmse": float(np.sqrt(_nanmean_or_nan((s_hat - S_true) ** 2))),
        "mc_sd_S_hat": float(np.std(s_hat[np.isfinite(s_hat)], ddof=1)) if np.count_nonzero(np.isfinite(s_hat)) > 1 else np.nan,
        "outer_reps": int(R),
        "garch_outer_reps": int(Rg) if GARCH_MLE_METHOD in cfg.methods else 0,
        "garch_fit_fail_count": int(garch_fail),
        "garch_fit_fail_rate": float(garch_fail / float(Rg)) if (GARCH_MLE_METHOD in cfg.methods and Rg > 0) else np.nan,
        "garch_regularized_count": int(garch_regularized),
        "garch_regularized_rate": float(garch_regularized / float(Rg)) if (GARCH_MLE_METHOD in cfg.methods and Rg > 0) else np.nan,
        "garch_fit_converged_rate": garch_fit_converged_rate,
        "garch_fit_attempts_mean": garch_fit_attempts_mean,
        "garch_fit_time_ms_mean": garch_fit_time_ms_mean,
    }
    return rows, diag, ci_rows


def run_cell(dgp: str, n: int, S_true: float, cfg: Config | dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, diag, _ = _run_cell_impl(dgp, n, S_true, cfg, ci_levels=None)
    return rows, diag


def _run_cell(spec: tuple[str, int, float, Config]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dgp, n, S_true, cfg = spec
    return run_cell(dgp, n, S_true, cfg)


def _run_cell_with_ci_sweep(
    spec: tuple[str, int, float, Config, tuple[float, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    dgp, n, S_true, cfg, ci_levels = spec
    return _run_cell_impl(dgp, n, S_true, cfg, ci_levels=ci_levels)


def _is_process_pool_bootstrap_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
        "start a new process before the current process has finished its bootstrapping phase" in message
        or "safe importing of main module" in message
    )


def _warn_parallel_fallback_once(exc: BaseException) -> None:
    global _parallel_fallback_warned
    if _parallel_fallback_warned:
        return
    warnings.warn(
        f"Process-pool execution failed ({exc.__class__.__name__}); falling back to serial execution.",
        RuntimeWarning,
        stacklevel=3,
    )
    _parallel_fallback_warned = True


def _run_partA_serial_specs(
    specs: list[tuple[str, int, float]],
    cfg: Config,
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    return [_run_cell((d, int(n), float(s), cfg)) for d, n, s in specs]


def _run_partA_with_ci_sweep_serial_specs(
    specs: list[tuple[str, int, float]],
    cfg: Config,
    levels: tuple[float, ...],
    progress_callback: Callable[[], None] | None,
) -> list[tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]]:
    out = []
    for d, n, s in specs:
        out.append(_run_cell_with_ci_sweep((d, int(n), float(s), cfg, levels)))
        if progress_callback is not None:
            progress_callback()
    return out


def run_partA(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = list(product(cfg.dgps, cfg.n_grid, cfg.S_grid))

    if cfg.max_workers <= 1 or len(specs) <= 1:
        out = _run_partA_serial_specs(specs, cfg)
    else:
        workers = min(int(cfg.max_workers), len(specs))
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                out = list(ex.map(_run_cell, [(d, int(n), float(s), cfg) for d, n, s in specs]))
        except BrokenProcessPool as exc:
            _warn_parallel_fallback_once(exc)
            out = _run_partA_serial_specs(specs, cfg)
        except RuntimeError as exc:
            if not _is_process_pool_bootstrap_error(exc):
                raise
            _warn_parallel_fallback_once(exc)
            out = _run_partA_serial_specs(specs, cfg)

    rows_m: list[dict[str, Any]] = []
    rows_d: list[dict[str, Any]] = []
    for m, d in out:
        rows_m.extend(m)
        rows_d.append(d)

    df_m = pd.DataFrame(rows_m).sort_values(["dgp", "n", "S_true", "method"]).reset_index(drop=True)
    df_d = pd.DataFrame(rows_d).sort_values(["dgp", "n", "S_true"]).reset_index(drop=True)
    return df_m, df_d


def run_partA_with_ci_sweep(
    cfg: Config,
    ci_levels: tuple[float, ...] | list[float],
    *,
    progress_callback: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    levels = normalize_ci_levels(ci_levels)
    specs = list(product(cfg.dgps, cfg.n_grid, cfg.S_grid))

    if cfg.max_workers <= 1 or len(specs) <= 1:
        out = _run_partA_with_ci_sweep_serial_specs(specs, cfg, levels, progress_callback)
    else:
        workers = min(int(cfg.max_workers), len(specs))
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                out = []
                for item in ex.map(_run_cell_with_ci_sweep, [(d, int(n), float(s), cfg, levels) for d, n, s in specs]):
                    out.append(item)
                    if progress_callback is not None:
                        progress_callback()
        except BrokenProcessPool as exc:
            _warn_parallel_fallback_once(exc)
            out = _run_partA_with_ci_sweep_serial_specs(specs, cfg, levels, progress_callback)
        except RuntimeError as exc:
            if not _is_process_pool_bootstrap_error(exc):
                raise
            _warn_parallel_fallback_once(exc)
            out = _run_partA_with_ci_sweep_serial_specs(specs, cfg, levels, progress_callback)

    rows_m: list[dict[str, Any]] = []
    rows_d: list[dict[str, Any]] = []
    rows_ci: list[dict[str, Any]] = []
    for m_rows, d_row, ci_rows in out:
        rows_m.extend(m_rows)
        rows_d.append(d_row)
        rows_ci.extend(ci_rows)

    df_m = pd.DataFrame(rows_m).sort_values(["dgp", "n", "S_true", "method"]).reset_index(drop=True)
    df_d = pd.DataFrame(rows_d).sort_values(["dgp", "n", "S_true"]).reset_index(drop=True)

    if rows_ci:
        df_ci = pd.DataFrame(rows_ci).sort_values(["dgp", "n", "S_true", "method", "ci_level"]).reset_index(drop=True)
    else:
        df_ci = pd.DataFrame(
            columns=[
                "dgp",
                "method",
                "n",
                "S_true",
                "ci_level",
                "outer_reps",
                "coverage",
                "avg_ci_length",
                "mc_se",
                "mc_lo",
                "mc_hi",
                "fit_fail_count",
                "fit_fail_rate",
            ]
        )
    return df_m, df_d, df_ci


def _env_scalar(name: str, default: Any, cast):
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else cast(raw.strip())


def _env_tuple(name: str, default: tuple[Any, ...], cast):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(cast(p.strip()) for p in raw.split(",") if p.strip())


def config_from_env() -> Config:
    max_workers_default = max(1, (os.cpu_count() or 2) - 1)
    return Config(
        seed=_env_scalar("SEED", 0, int),
        alpha=_env_scalar("ALPHA", 0.05, float),
        R=60025,
        R_garch=_env_scalar("R_GARCH", 1000, int),
        dgps=_env_tuple("DGPS", ("iid_normal", "garch11_t"), str),
        methods=(ANALYTIC_METHOD, GARCH_MLE_METHOD, GARCH_ORACLE_METHOD),
        n_grid=_env_tuple("N_GRID", (30, 60), int),
        S_grid=_env_tuple("S_GRID", (-0.5, 0.0, 0.5), float),
        g_alpha=_env_scalar("G_ALPHA", 0.05, float),
        g_beta=_env_scalar("G_BETA", 0.90, float),
        garch_dist=_env_scalar("GARCH_DIST", "t", str),
        nu=_env_scalar("NU", 7.0, float),
        burn=_env_scalar("BURN", 500, int),
        mle_maxiter_warm=_env_scalar("MLE_MAXITER_WARM", MAXITER_WARM, int),
        mle_maxiter_cold=_env_scalar("MLE_MAXITER_COLD", MAXITER_COLD, int),
        mle_tol=_env_scalar("MLE_TOL", MLE_TOL, float),
        max_workers=_env_scalar("MAX_WORKERS", max_workers_default, int),
    )


def smoke_garch11_runtime() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = Config(
        seed=0,
        R=2000,
        R_garch=50,
        dgps=("garch11_t",),
        methods=(ANALYTIC_METHOD, GARCH_MLE_METHOD, GARCH_ORACLE_METHOD),
        n_grid=(30,),
        S_grid=(0.0,),
        burn=50,
        max_workers=1,
    )
    df_m, df_d = run_partA(cfg)
    print(f"[smoke] df_m shape={df_m.shape}, df_d shape={df_d.shape}")
    return df_m, df_d


def smoke_garch11_oracle_analytic(seed: int = 0) -> dict[str, float]:
    start_t = time.perf_counter()
    cfg = Config(
        seed=int(seed),
        alpha=0.05,
        R=200,
        R_garch=1,
        dgps=("garch11_t",),
        methods=(GARCH_ORACLE_METHOD,),
        n_grid=(240,),
        S_grid=(0.5,),
        g_alpha=0.05,
        g_beta=0.90,
        garch_dist="t",
        nu=8.0,
        burn=1000,
        max_workers=1,
    )
    df_m, _ = run_partA(cfg)
    if df_m.empty:
        raise RuntimeError("Oracle smoke test returned no results.")
    row = df_m.iloc[0]
    required = row[["coverage_95", "mc_se", "mc_lo", "mc_hi", "se_cell", "omega_hat_cell"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(required)):
        raise RuntimeError("Oracle smoke test produced non-finite summary metrics.")
    elapsed_s = float(time.perf_counter() - start_t)
    if elapsed_s > 20.0:
        raise RuntimeError(f"Oracle smoke test too slow: {elapsed_s:.2f}s")
    coverage = float(row["coverage_95"])
    mc_se = float(row["mc_se"])
    print(f"[smoke-oracle] coverage={coverage:.4f}, mc_se={mc_se:.4f}, elapsed_s={elapsed_s:.2f}")
    return {"coverage": coverage, "mc_se": mc_se, "elapsed_s": elapsed_s}


if __name__ == "__main__":
    if os.environ.get("BSC_RUNTIME_SMOKE_ORACLE", "").strip() == "1":
        smoke_garch11_oracle_analytic()
        raise SystemExit(0)

    if os.environ.get("BSC_RUNTIME_SMOKE", "").strip() == "1":
        smoke_garch11_runtime()
        raise SystemExit(0)

    cfg = config_from_env()
    results, diagnostics = run_partA(cfg)
    out = Path(os.environ.get("EXPERIMENT_BSC_OUTPUT_DIR", "outputs/experiment_bsc")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "results_partA.csv", index=False)
    diagnostics.to_csv(out / "diagnostics_partA.csv", index=False)
    print(results)
    print(diagnostics)
