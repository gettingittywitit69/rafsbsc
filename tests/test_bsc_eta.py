from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import finance_data.bsc.api as api
import finance_data.bsc.eta as eta


def _tiny_cfg() -> api.Config:
    return api.default_config(
        R=40,
        R_garch=10,
        dgps=("iid_normal", "garch11_t"),
        methods=(api.ANALYTIC_METHOD, api.GARCH_MLE_METHOD, api.GARCH_ORACLE_METHOD),
        n_grid=(30,),
        S_grid=(0.0,),
        max_workers=1,
    )


def test_eta_cache_hit_returns_zero_without_pilot(tmp_path) -> None:
    cfg = _tiny_cfg()
    levels = (0.90, 0.95)
    scope = api._normalize_scope("main")
    run_dir = api._resolve_run_dir(tmp_path)
    cache_dir = run_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hash = api._cache_hash(cfg, scope, ci_levels=levels)

    for stem in ("results", "diagnostics", "ci_sweep"):
        (cache_dir / f"{scope}_{stem}_{cache_hash}.csv").write_text("", encoding="utf-8")

    estimate = eta.estimate_main_bundle_runtime(cfg, ci_levels=levels, output_dir=tmp_path)

    assert estimate.cache_hit is True
    assert estimate.eta_seconds == 0.0
    assert estimate.eta_low_seconds == 0.0
    assert estimate.eta_high_seconds == 0.0
    assert estimate.pilot_cells == 0
    assert "cache" in estimate.notes.lower()


def test_eta_pilot_path_returns_positive_eta_with_range(monkeypatch) -> None:
    cfg = _tiny_cfg()
    monkeypatch.setattr(eta, "_cache_hit_for_main", lambda *args, **kwargs: False)

    counter = {"n": 0}

    def fake_time(*args, **kwargs) -> float:
        counter["n"] += 1
        return 0.03 + 0.01 * counter["n"]

    monkeypatch.setattr(eta, "_time_single_cell", fake_time)

    estimate = eta.estimate_main_bundle_runtime(cfg, pilot_cells=3, pilot_R=10, pilot_R_garch=5)

    assert estimate.cache_hit is False
    assert estimate.pilot_cells == 2  # two cells in cfg grid
    assert estimate.eta_seconds > 0.0
    assert 0.0 <= estimate.eta_low_seconds <= estimate.eta_seconds <= estimate.eta_high_seconds


def test_eta_scales_monotonically_with_rep_counts(monkeypatch) -> None:
    monkeypatch.setattr(eta, "_cache_hit_for_main", lambda *args, **kwargs: False)
    monkeypatch.setattr(eta, "_time_single_cell", lambda *args, **kwargs: 0.05)

    cfg_small = api.default_config(
        R=50,
        R_garch=20,
        dgps=("iid_normal",),
        methods=(api.ANALYTIC_METHOD, api.GARCH_MLE_METHOD),
        n_grid=(30, 60),
        S_grid=(0.0,),
        max_workers=1,
    )
    cfg_large = api.default_config(
        R=200,
        R_garch=80,
        dgps=("iid_normal",),
        methods=(api.ANALYTIC_METHOD, api.GARCH_MLE_METHOD),
        n_grid=(30, 60),
        S_grid=(0.0,),
        max_workers=1,
    )

    est_small = eta.estimate_main_bundle_runtime(cfg_small, pilot_cells=2, pilot_R=10, pilot_R_garch=5)
    est_large = eta.estimate_main_bundle_runtime(cfg_large, pilot_cells=2, pilot_R=10, pilot_R_garch=5)
    assert est_large.eta_seconds > est_small.eta_seconds


def test_eta_uses_observed_parallel_probe_speedup(monkeypatch) -> None:
    monkeypatch.setattr(eta, "_cache_hit_for_main", lambda *args, **kwargs: False)
    monkeypatch.setattr(eta, "_time_single_cell", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(eta, "_time_pilot_specs_parallel", lambda *args, **kwargs: 0.08)

    cfg = api.default_config(
        R=100,
        R_garch=20,
        dgps=("iid_normal",),
        methods=(api.ANALYTIC_METHOD,),
        n_grid=(30, 60),
        S_grid=(0.0,),
        max_workers=4,
    )

    estimate = eta.estimate_main_bundle_runtime(cfg, pilot_cells=2, pilot_R=10, pilot_R_garch=10)

    assert estimate.eta_seconds == pytest.approx(1.0, rel=1e-9, abs=1e-9)
    assert estimate.eta_low_seconds == pytest.approx(1.0, rel=1e-9, abs=1e-9)
    assert estimate.eta_high_seconds == pytest.approx(1.0, rel=1e-9, abs=1e-9)
    assert "observed_speedup=2.00" in estimate.notes


def test_eta_parallel_probe_failure_uses_conservative_speedup(monkeypatch) -> None:
    monkeypatch.setattr(eta, "_cache_hit_for_main", lambda *args, **kwargs: False)
    monkeypatch.setattr(eta, "_time_single_cell", lambda *args, **kwargs: 0.1)

    def fail_probe(*args, **kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(eta, "_time_pilot_specs_parallel", fail_probe)

    cfg = api.default_config(
        R=100,
        R_garch=20,
        dgps=("iid_normal",),
        methods=(api.ANALYTIC_METHOD,),
        n_grid=(30, 60),
        S_grid=(0.0,),
        max_workers=4,
    )

    estimate = eta.estimate_main_bundle_runtime(cfg, pilot_cells=2, pilot_R=10, pilot_R_garch=10)

    assert estimate.eta_seconds == pytest.approx(2.0, rel=1e-9, abs=1e-9)
    assert estimate.eta_low_seconds == pytest.approx(2.0, rel=1e-9, abs=1e-9)
    assert estimate.eta_high_seconds == pytest.approx(2.0, rel=1e-9, abs=1e-9)
    assert "parallel probe failed" in estimate.notes
    assert "conservative speedup=1.00" in estimate.notes


def test_eta_cli_prints_estimate(monkeypatch, capsys) -> None:
    fake = eta.RuntimeEstimate(
        eta_seconds=123.0,
        eta_low_seconds=100.0,
        eta_high_seconds=160.0,
        cache_hit=False,
        pilot_cells=4,
        pilot_R=20,
        pilot_R_garch=10,
        notes="test note",
    )
    monkeypatch.setattr(eta, "estimate_main_bundle_runtime", lambda *args, **kwargs: fake)

    rc = eta.main(["--pilot-cells", "4"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "ETA:" in out
    assert "range" in out
    assert "pilot_cells=4" in out
