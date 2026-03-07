from __future__ import annotations

import numpy as np

EPS = 1e-12


def h2_from_innov(dist: str, nu: float | None) -> float:
    """
    Innovation 4th-moment ratio h2 = E[eps^4] / E[eps^2]^2 for unit-variance eps.
    """
    name = str(dist).strip().lower()
    if name in {"normal", "gaussian"}:
        return 3.0
    if name in {"t", "student_t", "student-t"}:
        if nu is None:
            raise ValueError("nu is required for t innovations and must satisfy nu > 4.")
        nu_f = float(nu)
        if (not np.isfinite(nu_f)) or nu_f <= 4.0:
            raise ValueError("nu must be finite and > 4 for t innovations.")
        return float(3.0 * (nu_f - 2.0) / (nu_f - 4.0))
    raise ValueError(f"Unsupported innovation distribution: {dist}")


def omega_garch_closed_form(S_hat: float | np.ndarray, alpha1: float, beta: float, h2: float) -> float | np.ndarray:
    """
    Closed-form long-run variance for Sharpe under symmetric GARCH(1,1):
      Omega_G = 1 + (S^2/4) * ((h2-1)(1+gamma)(1-beta)^2)/(d(1-gamma)),
      gamma = alpha1 + beta, d = 1 - gamma^2 - (h2-1)alpha1^2.
    """
    alpha1_f = float(alpha1)
    beta_f = float(beta)
    h2_f = float(h2)
    gamma = alpha1_f + beta_f
    d = 1.0 - gamma**2 - (h2_f - 1.0) * alpha1_f**2

    if (not np.isfinite(d)) or d <= 0.0 or (not np.isfinite(gamma)) or (1.0 - gamma) <= 0.0:
        arr = np.asarray(S_hat, dtype=float)
        if arr.ndim == 0:
            return float("nan")
        return np.full(arr.shape, np.nan, dtype=float)

    factor = ((h2_f - 1.0) * (1.0 + gamma) * (1.0 - beta_f) ** 2) / (d * (1.0 - gamma))
    omega = 1.0 + (np.asarray(S_hat, dtype=float) ** 2 / 4.0) * factor

    if np.ndim(omega) == 0:
        omega_f = float(omega)
        return omega_f if np.isfinite(omega_f) and omega_f > 0.0 else float("nan")

    omega = np.asarray(omega, dtype=float)
    bad = (~np.isfinite(omega)) | (omega <= 0.0)
    omega[bad] = np.nan
    return omega


def simulate_garch11(
    T: int,
    mu: float,
    alpha0: float,
    alpha1: float,
    beta: float,
    dist: str,
    nu: float | None,
    burn: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate one GARCH(1,1) path x_t = mu + sqrt(sigma2_t) * eps_t.
    """
    T = int(T)
    burn = int(burn)
    if T < 1:
        raise ValueError("T must be >= 1.")
    if burn < 0:
        raise ValueError("burn must be >= 0.")

    alpha0_f = float(alpha0)
    alpha1_f = float(alpha1)
    beta_f = float(beta)
    if alpha1_f < 0.0 or beta_f < 0.0:
        raise ValueError("alpha1 and beta must be >= 0.")

    gamma = alpha1_f + beta_f
    sigma2_uncond = float(alpha0_f / (1.0 - gamma)) if gamma < 1.0 else 1.0
    sigma2_prev = float(max(sigma2_uncond, EPS))
    x_prev = float(mu)

    total = T + burn
    name = str(dist).strip().lower()
    if name in {"normal", "gaussian"}:
        eps = rng.standard_normal(total)
    elif name in {"t", "student_t", "student-t"}:
        if nu is None:
            raise ValueError("nu is required for t innovations.")
        nu_f = float(nu)
        if (not np.isfinite(nu_f)) or nu_f <= 2.0:
            raise ValueError("nu must be finite and > 2 to standardize t innovations to unit variance.")
        eps = rng.standard_t(df=nu_f, size=total) * np.sqrt((nu_f - 2.0) / nu_f)
    else:
        raise ValueError(f"Unsupported innovation distribution: {dist}")

    out = np.empty(total, dtype=float)
    for t in range(total):
        sigma2_t = alpha0_f + alpha1_f * (x_prev - mu) ** 2 + beta_f * sigma2_prev
        sigma2_t = float(max(sigma2_t, EPS))
        x_t = float(mu + np.sqrt(sigma2_t) * eps[t])
        out[t] = x_t
        x_prev = x_t
        sigma2_prev = sigma2_t
    return out[burn:]
