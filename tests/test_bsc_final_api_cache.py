from pathlib import Path
import sys

import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import finance_data.bsc.api as api


def _sample_outputs(coverage: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.DataFrame(
        [
            {
                "dgp": "iid_normal",
                "method": api.ANALYTIC_METHOD,
                "n": 30,
                "S_true": 0.0,
                "coverage_95": float(coverage),
                "reject_rate_H0_S_eq_0": 0.05,
            }
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "dgp": "iid_normal",
                "n": 30,
                "S_true": 0.0,
                "bias": 0.0,
                "rmse": 0.1,
            }
        ]
    )
    return results, diagnostics


def _sample_outputs_with_ci_sweep(ci_levels: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results, diagnostics = _sample_outputs()
    ci_rows = [
        {
            "dgp": "iid_normal",
            "method": api.ANALYTIC_METHOD,
            "n": 30,
            "S_true": 0.0,
            "ci_level": float(level),
            "outer_reps": 10,
            "coverage": 0.95,
            "avg_ci_length": 0.50 + 0.10 * float(level),
            "mc_se": 0.01,
            "mc_lo": 0.93,
            "mc_hi": 0.97,
            "fit_fail_count": 0,
            "fit_fail_rate": 0.0,
        }
        for level in ci_levels
    ]
    return results, diagnostics, pd.DataFrame(ci_rows)


def _tiny_cfg() -> api.Config:
    return api.default_config(
        R=10,
        R_garch=5,
        dgps=("iid_normal",),
        methods=(api.ANALYTIC_METHOD,),
        n_grid=(30,),
        S_grid=(0.0,),
        max_workers=1,
    )


def test_run_cached_miss_writes_cache_and_meta(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "run_partA", lambda cfg: _sample_outputs())

    results, diagnostics, meta = api.run_cached(_tiny_cfg(), output_dir=tmp_path)

    assert meta["cache_hit"] is False
    assert set(meta.keys()) == {
        "cache_hit",
        "cache_hash",
        "scope",
        "run_dir",
        "cache_dir",
        "results_path",
        "diagnostics_path",
        "config_path",
    }
    assert Path(meta["results_path"]).exists()
    assert Path(meta["diagnostics_path"]).exists()
    assert Path(meta["config_path"]).exists()
    assert not results.empty
    assert not diagnostics.empty


def test_run_cached_hit_skips_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "run_partA", lambda cfg: _sample_outputs())
    results_first, diagnostics_first, _ = api.run_cached(_tiny_cfg(), output_dir=tmp_path)

    def explode(_cfg):
        raise RuntimeError("run_partA should not run on cache hit")

    monkeypatch.setattr(api, "run_partA", explode)
    results_second, diagnostics_second, meta_second = api.run_cached(_tiny_cfg(), output_dir=tmp_path)

    assert meta_second["cache_hit"] is True
    assert_frame_equal(results_first, results_second, check_dtype=False)
    assert_frame_equal(diagnostics_first, diagnostics_second, check_dtype=False)


def test_run_cached_scope_isolated(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "run_partA", lambda cfg: _sample_outputs())

    _, _, main_meta = api.run_cached(_tiny_cfg(), output_dir=tmp_path, scope="main")
    _, _, oracle_meta = api.run_cached(_tiny_cfg(), output_dir=tmp_path, scope="oracle")

    assert main_meta["cache_hash"] != oracle_meta["cache_hash"]
    assert main_meta["results_path"] != oracle_meta["results_path"]
    assert Path(main_meta["results_path"]).exists()
    assert Path(oracle_meta["results_path"]).exists()


def test_run_cached_force_rerun_bypasses_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "run_partA", lambda cfg: _sample_outputs(coverage=0.91))
    api.run_cached(_tiny_cfg(), output_dir=tmp_path)

    calls = {"n": 0}

    def rerun(_cfg):
        calls["n"] += 1
        return _sample_outputs(coverage=0.99)

    monkeypatch.setattr(api, "run_partA", rerun)
    results, _, meta = api.run_cached(_tiny_cfg(), output_dir=tmp_path, force_rerun=True)

    assert calls["n"] == 1
    assert meta["cache_hit"] is False
    assert float(results.loc[0, "coverage_95"]) == 0.99


def test_run_cached_hash_changes_on_config_change(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(api, "run_partA", lambda cfg: _sample_outputs())

    cfg_a = _tiny_cfg()
    cfg_b = api.default_config(
        R=20,
        R_garch=5,
        dgps=("iid_normal",),
        methods=(api.ANALYTIC_METHOD,),
        n_grid=(30,),
        S_grid=(0.0,),
        max_workers=1,
    )
    _, _, meta_a = api.run_cached(cfg_a, output_dir=tmp_path, scope="main")
    _, _, meta_b = api.run_cached(cfg_b, output_dir=tmp_path, scope="main")

    assert meta_a["cache_hash"] != meta_b["cache_hash"]


def test_run_ci_sweep_cached_hash_changes_when_ci_levels_change(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        api,
        "run_partA_with_ci_sweep",
        lambda cfg, ci_levels: _sample_outputs_with_ci_sweep(tuple(float(x) for x in ci_levels)),
    )
    cfg = _tiny_cfg()

    _, _, ci_a, meta_a = api.run_ci_sweep_cached(cfg, ci_levels=(0.90, 0.95), output_dir=tmp_path, scope="main")
    _, _, ci_b, meta_b = api.run_ci_sweep_cached(cfg, ci_levels=(0.90, 0.99), output_dir=tmp_path, scope="main")

    assert not ci_a.empty
    assert not ci_b.empty
    assert meta_a["cache_hash"] != meta_b["cache_hash"]
    assert meta_a["ci_sweep_path"] != meta_b["ci_sweep_path"]
    assert Path(meta_a["ci_sweep_path"]).exists()
    assert Path(meta_b["ci_sweep_path"]).exists()
