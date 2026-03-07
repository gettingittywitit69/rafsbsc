from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_data.bsc import garch_oracle as go
from finance_data.bsc import runtime as rt


def test_run_cell_accepts_dict_config_for_zero_sharpe() -> None:
    cfg = rt.Config(
        R=40,
        R_garch=10,
        dgps=("iid_normal",),
        methods=(rt.ANALYTIC_METHOD,),
        n_grid=(30,),
        S_grid=(0.0,),
        max_workers=1,
    )

    rows, diag = rt.run_cell("iid_normal", 30, 0.0, asdict(cfg))

    assert len(rows) == 1
    assert rows[0]["S_true"] == 0.0
    assert rows[0]["outer_reps"] == 40
    assert diag["S_true"] == 0.0
    assert diag["outer_reps"] == 40


def test_garch_metrics_are_nan_when_all_fits_fail(monkeypatch) -> None:
    cfg = rt.Config(
        R=20,
        R_garch=5,
        dgps=("iid_normal",),
        methods=(rt.GARCH_MLE_METHOD,),
        n_grid=(30,),
        S_grid=(0.0,),
        max_workers=1,
    )

    def always_fail(*args, **kwargs):
        return {"ok": False}

    monkeypatch.setattr(rt, "_fit_garch11", always_fail)

    rows, diag = rt.run_cell("iid_normal", 30, 0.0, cfg)

    assert len(rows) == 1
    assert rows[0]["fit_fail_count"] == 5
    assert rows[0]["fit_fail_rate"] == 1.0
    assert str(rows[0]["coverage_95"]) == "nan"
    assert str(rows[0]["reject_rate_H0_S_eq_0"]) == "nan"
    assert str(rows[0]["se_cell"]) == "nan"
    assert diag["garch_fit_fail_count"] == 5
    assert diag["garch_fit_fail_rate"] == 1.0


def test_fit_garch11_uses_moment_fallback_when_optimizer_raises(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    monkeypatch.setattr(rt.sharpe_mc, "fit_candidate", explode)

    x = np.linspace(-1.0, 1.0, 20)
    fit = rt._fit_garch11(x, dist="student_t", start=None, maxiter=rt.MAXITER_COLD, fallback_nu=7.0)

    assert fit["ok"] is True
    assert fit["regularized"] is True
    assert fit["params_vec"] is None
    assert np.isfinite(rt.omega_garch_plugin(0.0, alpha1=fit["alpha1"], beta=fit["beta"], h2=fit["h2"]))


def test_small_sample_garch_run_cell_returns_finite_plugin_metrics() -> None:
    cfg = rt.Config(
        R=50,
        R_garch=20,
        dgps=("garch11_t",),
        methods=(rt.GARCH_MLE_METHOD,),
        n_grid=(20,),
        S_grid=(0.0,),
        max_workers=1,
    )

    rows, diag = rt.run_cell("garch11_t", 20, 0.0, cfg)

    assert len(rows) == 1
    assert rows[0]["fit_fail_count"] == 0
    assert rows[0]["regularized_count"] >= 0
    assert np.isfinite(rows[0]["se_cell"])
    assert np.isfinite(rows[0]["omega_hat_cell"])
    assert diag["garch_fit_fail_count"] == 0
    assert np.isfinite(diag["garch_regularized_rate"])


def test_oracle_row_has_mc_error_bars_and_never_fits(monkeypatch) -> None:
    cfg = rt.Config(
        R=60,
        R_garch=5,
        dgps=("garch11_t",),
        methods=(rt.GARCH_ORACLE_METHOD,),
        n_grid=(30,),
        S_grid=(0.5,),
        g_alpha=0.05,
        g_beta=0.90,
        garch_dist="t",
        nu=8.0,
        burn=200,
        max_workers=1,
    )

    def explode(*args, **kwargs):
        raise RuntimeError("fit_candidate must not be called for oracle-only runs")

    monkeypatch.setattr(rt.sharpe_mc, "fit_candidate", explode)
    rows, _ = rt.run_cell("garch11_t", 30, 0.5, cfg)

    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == rt.GARCH_ORACLE_METHOD
    assert np.isfinite(row["coverage_95"])
    assert np.isfinite(row["mc_se"])
    assert np.isfinite(row["mc_lo"])
    assert np.isfinite(row["mc_hi"])


def test_garch_oracle_helpers_match_closed_form_and_guards() -> None:
    h2_t = go.h2_from_innov("t", 8.0)
    h2_n = go.h2_from_innov("normal", None)
    assert np.isclose(h2_t, 4.5)
    assert np.isclose(h2_n, 3.0)

    omega = go.omega_garch_closed_form(0.5, 0.05, 0.90, h2_t)
    assert np.isfinite(omega)

    # Invalid denominator case -> NaN (gamma >= 1).
    bad = go.omega_garch_closed_form(0.5, 0.10, 0.95, h2_t)
    assert str(bad) == "nan"


def test_run_partA_with_ci_sweep_has_expected_grid_rows() -> None:
    cfg = rt.Config(
        R=40,
        R_garch=10,
        dgps=("iid_normal", "garch11_t"),
        methods=(rt.ANALYTIC_METHOD, rt.GARCH_MLE_METHOD, rt.GARCH_ORACLE_METHOD),
        n_grid=(20,),
        S_grid=(0.0,),
        burn=100,
        max_workers=1,
    )
    ci_levels = (0.90, 0.95, 0.99)

    results, diagnostics, ci_sweep = rt.run_partA_with_ci_sweep(cfg, ci_levels)

    assert not results.empty
    assert not diagnostics.empty
    assert not ci_sweep.empty

    expected = {
        (row.dgp, row.method, int(row.n), float(row.S_true), float(level))
        for row in results.itertuples(index=False)
        for level in ci_levels
    }
    actual = {
        (row.dgp, row.method, int(row.n), float(row.S_true), float(row.ci_level))
        for row in ci_sweep.itertuples(index=False)
    }

    assert len(ci_sweep) == len(expected)
    assert actual == expected


def test_run_partA_with_ci_sweep_invokes_progress_callback_for_each_cell_single_worker() -> None:
    cfg = rt.Config(
        R=20,
        R_garch=5,
        dgps=("iid_normal",),
        methods=(rt.ANALYTIC_METHOD,),
        n_grid=(20, 30),
        S_grid=(0.0, 0.5),
        max_workers=1,
    )
    ticks = {"count": 0}

    def on_cell_complete() -> None:
        ticks["count"] += 1

    rt.run_partA_with_ci_sweep(cfg, (0.95,), progress_callback=on_cell_complete)

    expected_cells = len(cfg.dgps) * len(cfg.n_grid) * len(cfg.S_grid)
    assert ticks["count"] == expected_cells


def test_run_partA_with_ci_sweep_invokes_progress_callback_for_each_cell_multi_worker() -> None:
    cfg = rt.Config(
        R=20,
        R_garch=5,
        dgps=("iid_normal",),
        methods=(rt.ANALYTIC_METHOD,),
        n_grid=(20, 30),
        S_grid=(0.0,),
        max_workers=2,
    )
    ticks = {"count": 0}

    def on_cell_complete() -> None:
        ticks["count"] += 1

    rt.run_partA_with_ci_sweep(cfg, (0.95,), progress_callback=on_cell_complete)

    expected_cells = len(cfg.dgps) * len(cfg.n_grid) * len(cfg.S_grid)
    assert ticks["count"] == expected_cells


def test_run_partA_parallel_failure_falls_back_to_serial(monkeypatch) -> None:
    cfg = rt.Config(
        R=20,
        R_garch=5,
        dgps=("iid_normal",),
        methods=(rt.ANALYTIC_METHOD,),
        n_grid=(20, 30),
        S_grid=(0.0,),
        max_workers=2,
    )
    calls = {"count": 0}

    def fake_run_cell(spec):
        dgp, n, s_true, _cfg = spec
        calls["count"] += 1
        return (
            [
                {
                    "dgp": dgp,
                    "method": rt.ANALYTIC_METHOD,
                    "n": int(n),
                    "S_true": float(s_true),
                    "coverage_95": 0.95,
                    "reject_rate_H0_S_eq_0": 0.05,
                }
            ],
            {"dgp": dgp, "n": int(n), "S_true": float(s_true), "bias": 0.0, "rmse": 0.1},
        )

    class BrokenPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, iterable):
            raise BrokenProcessPool("pool crashed")

    monkeypatch.setattr(rt, "ProcessPoolExecutor", BrokenPool)
    monkeypatch.setattr(rt, "_run_cell", fake_run_cell)
    monkeypatch.setattr(rt, "_parallel_fallback_warned", False)

    with pytest.warns(RuntimeWarning, match="falling back to serial"):
        results, diagnostics = rt.run_partA(cfg)

    expected_cells = len(cfg.dgps) * len(cfg.n_grid) * len(cfg.S_grid)
    assert calls["count"] == expected_cells
    assert len(results) == expected_cells
    assert len(diagnostics) == expected_cells


def test_run_partA_with_ci_sweep_parallel_failure_falls_back_to_serial_and_ticks(monkeypatch) -> None:
    cfg = rt.Config(
        R=20,
        R_garch=5,
        dgps=("iid_normal",),
        methods=(rt.ANALYTIC_METHOD,),
        n_grid=(20, 30),
        S_grid=(0.0,),
        max_workers=2,
    )
    calls = {"count": 0}
    ticks = {"count": 0}

    def fake_run_cell_with_ci_sweep(spec):
        dgp, n, s_true, _cfg, levels = spec
        calls["count"] += 1
        ci_level = float(levels[0]) if levels else 0.95
        return (
            [
                {
                    "dgp": dgp,
                    "method": rt.ANALYTIC_METHOD,
                    "n": int(n),
                    "S_true": float(s_true),
                    "coverage_95": 0.95,
                    "reject_rate_H0_S_eq_0": 0.05,
                }
            ],
            {"dgp": dgp, "n": int(n), "S_true": float(s_true), "bias": 0.0, "rmse": 0.1},
            [
                {
                    "dgp": dgp,
                    "method": rt.ANALYTIC_METHOD,
                    "n": int(n),
                    "S_true": float(s_true),
                    "ci_level": ci_level,
                    "outer_reps": 20,
                    "coverage": 0.95,
                    "avg_ci_length": 0.2,
                    "mc_se": 0.01,
                    "mc_lo": 0.93,
                    "mc_hi": 0.97,
                    "fit_fail_count": 0,
                    "fit_fail_rate": 0.0,
                }
            ],
        )

    class BrokenPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, iterable):
            raise BrokenProcessPool("pool crashed")

    def on_cell_complete() -> None:
        ticks["count"] += 1

    monkeypatch.setattr(rt, "ProcessPoolExecutor", BrokenPool)
    monkeypatch.setattr(rt, "_run_cell_with_ci_sweep", fake_run_cell_with_ci_sweep)
    monkeypatch.setattr(rt, "_parallel_fallback_warned", False)

    with pytest.warns(RuntimeWarning, match="falling back to serial"):
        results, diagnostics, ci_sweep = rt.run_partA_with_ci_sweep(cfg, (0.95,), progress_callback=on_cell_complete)

    expected_cells = len(cfg.dgps) * len(cfg.n_grid) * len(cfg.S_grid)
    assert calls["count"] == expected_cells
    assert ticks["count"] == expected_cells
    assert len(results) == expected_cells
    assert len(diagnostics) == expected_cells
    assert len(ci_sweep) == expected_cells


def test_ci_sweep_avg_ci_length_is_weakly_increasing_in_ci_level() -> None:
    cfg = rt.Config(
        R=80,
        R_garch=20,
        dgps=("iid_normal",),
        methods=(rt.ANALYTIC_METHOD,),
        n_grid=(30,),
        S_grid=(0.0, 0.5),
        max_workers=1,
    )
    ci_levels = (0.90, 0.95, 0.975, 0.99)

    _, _, ci_sweep = rt.run_partA_with_ci_sweep(cfg, ci_levels)

    for _, group in ci_sweep.groupby(["dgp", "method", "n", "S_true"], sort=False):
        ordered = group.sort_values("ci_level")
        lengths = ordered["avg_ci_length"].to_numpy(dtype=float)
        finite = lengths[np.isfinite(lengths)]
        if finite.size > 1:
            assert np.all(np.diff(finite) >= -1e-12)
