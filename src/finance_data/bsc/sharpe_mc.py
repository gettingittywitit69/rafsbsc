from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
import numpy as np

from arch.univariate import ConstantMean, GARCH, Normal, StudentsT
from arch.utility.exceptions import StartingValueWarning


def _ensure_1d_finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 10:
        raise ValueError("x must be 1d with at least 10 observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains non-finite values")
    return arr


def _params_array(res) -> np.ndarray:
    try:
        params = np.asarray(res.params, dtype=float).reshape(-1)
    except Exception:
        return np.empty(0, dtype=float)
    return params


def _fit_succeeded(res) -> bool:
    """
    Robust success check for arch fits.

    Preference order:
      1) convergence_flag == 0
      2) optimization_result.success
      3) finite params and finite loglikelihood (if present)
    """
    params = _params_array(res)
    if params.size == 0 or not np.all(np.isfinite(params)):
        return False

    flag = getattr(res, "convergence_flag", None)
    if flag is not None:
        try:
            return int(flag) == 0
        except Exception:
            return False

    opt = getattr(res, "optimization_result", None)
    if opt is not None and hasattr(opt, "success"):
        try:
            return bool(opt.success)
        except Exception:
            return False

    ll = getattr(res, "loglikelihood", None)
    if ll is not None:
        try:
            return bool(np.isfinite(float(ll)))
        except Exception:
            return False
    return True


def _default_starting_values(model) -> np.ndarray:
    # Build the same component-wise default used by arch fit(..., starting_values=None).
    model._adjust_sample(None, None)
    mean_sv = np.asarray(model.starting_values(), dtype=float).reshape(-1)
    resids = np.asarray(model.resids(mean_sv), dtype=float).reshape(-1)

    sv_vol = np.asarray(model.volatility.starting_values(resids), dtype=float).reshape(-1)
    sigma2 = np.zeros(resids.shape[0], dtype=float)
    backcast = model.volatility.backcast(resids)
    var_bounds = model.volatility.variance_bounds(resids)
    model.volatility.compute_variance(sv_vol, resids, sigma2, backcast, var_bounds)
    std_resids = resids / np.sqrt(np.maximum(sigma2, np.finfo(float).tiny))
    sv_dist = np.asarray(model.distribution.starting_values(std_resids), dtype=float).reshape(-1)

    return np.asarray(np.hstack([mean_sv, sv_vol, sv_dist]), dtype=float)


def _starting_values_vector(model, starting_values):
    if starting_values is None:
        return None
    if isinstance(starting_values, Mapping):
        defaults = _default_starting_values(model)
        names = list(model._all_parameter_names())
        if defaults.size < len(names):
            defaults = np.pad(defaults, (0, len(names) - defaults.size), constant_values=np.nan)
        elif defaults.size > len(names):
            defaults = defaults[: len(names)]

        vec = np.asarray(defaults, dtype=float).copy()
        idx = {str(name): i for i, name in enumerate(names)}
        for key, value in starting_values.items():
            i = idx.get(str(key))
            if i is None:
                continue
            try:
                v = float(value)
            except Exception:
                continue
            if np.isfinite(v):
                vec[i] = v
        return vec
    return np.asarray(starting_values, dtype=float).reshape(-1)


def _attach_fit_meta(res, *, fit_converged: bool, fit_attempts: int, fit_time_ms: float) -> None:
    try:
        setattr(
            res,
            "_fit_meta",
            {
                "fit_converged": bool(fit_converged),
                "fit_attempts": int(fit_attempts),
                "fit_time_ms": float(fit_time_ms),
            },
        )
    except Exception:
        pass


def fit_candidate(
    x: np.ndarray,
    name: str,
    *,
    starting_values: np.ndarray | dict[str, float] | None = None,
    maxiter: int = 200,
    tol: float = 1e-6,
):
    """
    Minimal arch fitter used by bsc_final_runtime.

    Supported names:
      - "garch11_normal"
      - "garch11_t"
    """
    x = _ensure_1d_finite(x)

    if name == "garch11_normal":
        model = ConstantMean(x)
        model.volatility = GARCH(p=1, o=0, q=1)
        model.distribution = Normal()
    elif name == "garch11_t":
        model = ConstantMean(x)
        model.volatility = GARCH(p=1, o=0, q=1)
        model.distribution = StudentsT()
    else:
        raise ValueError(f"Unknown model name: {name}")

    fit_kwargs = {
        "update_freq": 0,
        "disp": "off",
        "show_warning": False,
        "tol": float(tol),
        "options": {"maxiter": int(maxiter), "disp": False},
    }

    sv = _starting_values_vector(model, starting_values)
    attempt_starts = [sv] if sv is not None else [None]
    if sv is not None:
        attempt_starts.append(None)

    start_t = time.perf_counter()
    last_exc = None
    last_res = None
    last_params = np.empty(0, dtype=float)

    for attempt_i, sv_i in enumerate(attempt_starts, start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", StartingValueWarning)
            try:
                if sv_i is None:
                    res = model.fit(**fit_kwargs)
                else:
                    res = model.fit(starting_values=sv_i, **fit_kwargs)
            except Exception as exc:
                last_exc = exc
                continue

        params = _params_array(res)
        if _fit_succeeded(res):
            fit_time_ms = (time.perf_counter() - start_t) * 1000.0
            _attach_fit_meta(res, fit_converged=True, fit_attempts=attempt_i, fit_time_ms=fit_time_ms)
            return model, res, params

        last_res = res
        last_params = params

    fit_time_ms = (time.perf_counter() - start_t) * 1000.0
    if last_res is not None:
        _attach_fit_meta(
            last_res,
            fit_converged=False,
            fit_attempts=len(attempt_starts),
            fit_time_ms=fit_time_ms,
        )
        return model, last_res, last_params
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("ARCH fit failed without a result")


def smoke_test_fit_candidate(seed: int = 0, n: int = 300) -> dict[str, bool]:
    x = np.random.default_rng(int(seed)).standard_normal(int(n))
    out: dict[str, bool] = {}
    for name in ("garch11_normal", "garch11_t"):
        _, res, params = fit_candidate(x, name)
        out[name] = bool(_fit_succeeded(res) and params.size > 0 and np.all(np.isfinite(params)))
    return out


if __name__ == "__main__":
    summary = smoke_test_fit_candidate()
    print(summary)
    if not all(summary.values()):
        raise SystemExit("smoke_test_fit_candidate failed")
