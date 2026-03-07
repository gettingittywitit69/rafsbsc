from pathlib import Path
import importlib
import sys
import warnings

import pandas as pd
from plotly.basedatatypes import BaseFigure
from plotly.graph_objs import Figure
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import finance_data.bsc.api as api


def _sample_oracle_results() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for n in (12, 36, 60):
        for s_true in (-0.5, 0.0, 0.5):
            coverage = 0.94 + 0.005 * (n >= 36) - 0.005 * abs(s_true)
            rows.append(
                {
                    "dgp": "garch11_t",
                    "method": api.GARCH_ORACLE_METHOD,
                    "n": n,
                    "S_true": s_true,
                    "coverage_95": coverage,
                    "mc_lo": coverage - 0.01,
                    "mc_hi": coverage + 0.01,
                    "se_cell": 0.10 + 0.01 * (n == 12),
                }
            )
    return pd.DataFrame(rows)


def test_run_main_bundle_returns_dataclass_payload(monkeypatch) -> None:
    results = pd.DataFrame(
        [
            {
                "dgp": "iid_normal",
                "method": api.ANALYTIC_METHOD,
                "n": 30,
                "S_true": 0.0,
                "coverage_95": 0.95,
                "reject_rate_H0_S_eq_0": 0.05,
            }
        ]
    )
    diagnostics = pd.DataFrame([{"dgp": "iid_normal", "n": 30, "S_true": 0.0, "bias": 0.0, "rmse": 0.1}])
    ci_sweep = pd.DataFrame([{"dgp": "iid_normal", "method": api.ANALYTIC_METHOD, "n": 30, "S_true": 0.0}])
    cache_meta = {"cache_hit": True, "cache_hash": "abc123"}

    monkeypatch.setattr(api, "run_ci_sweep_cached", lambda *args, **kwargs: (results, diagnostics, ci_sweep, cache_meta))
    monkeypatch.setattr(api, "plot_all", lambda *args, **kwargs: {"coverage_95": Figure()})

    bundle = api.run_main_bundle(
        api.default_config(R=10, R_garch=5, n_grid=(30,), S_grid=(0.0,), max_workers=1),
        export_png=False,
    )

    assert isinstance(bundle, api.MainBundle)
    assert bundle.results.equals(results)
    assert bundle.diagnostics.equals(diagnostics)
    assert bundle.ci_sweep.equals(ci_sweep)
    assert bundle.cache_meta == cache_meta
    assert "coverage_95" in bundle.figures


def test_run_main_bundle_progress_true_tracks_stage_and_sub_bars(monkeypatch) -> None:
    cfg = api.default_config(R=10, R_garch=5, n_grid=(30,), S_grid=(0.0,), max_workers=1)
    results = pd.DataFrame(
        [
            {
                "dgp": "iid_normal",
                "method": api.ANALYTIC_METHOD,
                "n": 30,
                "S_true": 0.0,
                "coverage_95": 0.95,
                "reject_rate_H0_S_eq_0": 0.05,
            }
        ]
    )
    diagnostics = pd.DataFrame([{"dgp": "iid_normal", "n": 30, "S_true": 0.0, "bias": 0.0, "rmse": 0.1}])
    ci_sweep = pd.DataFrame([{"dgp": "iid_normal", "method": api.ANALYTIC_METHOD, "n": 30, "S_true": 0.0}])
    cache_meta = {"cache_hit": False, "cache_hash": "abc123"}

    def fake_run_ci_sweep_cached(*args, progress_callback=None, **kwargs):
        assert progress_callback is not None
        for _ in range(2):
            progress_callback()
        return results, diagnostics, ci_sweep, cache_meta

    def fake_plot_all(*args, progress_callback=None, **kwargs):
        assert progress_callback is not None
        for _ in range(5):
            progress_callback()
        return {"coverage_95": Figure()}

    bars: list[object] = []

    class FakeBar:
        def __init__(self, *, total=None, **kwargs):
            self.total = total
            self.n = 0
            self.kwargs = kwargs
            self.closed = False
            bars.append(self)

        def update(self, n=1):
            self.n += n

        def close(self):
            self.closed = True

    monkeypatch.setattr(api, "_tqdm", FakeBar)
    monkeypatch.setattr(api, "run_ci_sweep_cached", fake_run_ci_sweep_cached)
    monkeypatch.setattr(api, "plot_all", fake_plot_all)

    bundle = api.run_main_bundle(cfg, progress=True, export_png=False)

    assert isinstance(bundle, api.MainBundle)
    assert len(bars) == 3
    stage_bar, compute_bar, plot_bar = bars
    assert stage_bar.total == 2
    assert stage_bar.n == 2
    assert compute_bar.total == len(cfg.dgps) * len(cfg.n_grid) * len(cfg.S_grid)
    assert compute_bar.n == compute_bar.total
    assert plot_bar.total == 5
    assert plot_bar.n == 5
    assert all(getattr(bar, "closed") for bar in bars)


def test_run_main_bundle_progress_none_stays_silent_in_ci(monkeypatch) -> None:
    cfg = api.default_config(R=10, R_garch=5, n_grid=(30,), S_grid=(0.0,), max_workers=1)
    results = pd.DataFrame(
        [
            {
                "dgp": "iid_normal",
                "method": api.ANALYTIC_METHOD,
                "n": 30,
                "S_true": 0.0,
                "coverage_95": 0.95,
                "reject_rate_H0_S_eq_0": 0.05,
            }
        ]
    )
    diagnostics = pd.DataFrame([{"dgp": "iid_normal", "n": 30, "S_true": 0.0, "bias": 0.0, "rmse": 0.1}])
    ci_sweep = pd.DataFrame([{"dgp": "iid_normal", "method": api.ANALYTIC_METHOD, "n": 30, "S_true": 0.0}])
    cache_meta = {"cache_hit": True, "cache_hash": "abc123"}
    captured: dict[str, object | None] = {"compute_cb": None, "plot_cb": None}

    def fake_run_ci_sweep_cached(*args, progress_callback=None, **kwargs):
        captured["compute_cb"] = progress_callback
        return results, diagnostics, ci_sweep, cache_meta

    def fake_plot_all(*args, progress_callback=None, **kwargs):
        captured["plot_cb"] = progress_callback
        return {"coverage_95": Figure()}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("tqdm should not be constructed when progress auto-disables in CI")

    monkeypatch.setenv("CI", "1")
    monkeypatch.setattr(api, "_tqdm", fail_if_called)
    monkeypatch.setattr(api, "run_ci_sweep_cached", fake_run_ci_sweep_cached)
    monkeypatch.setattr(api, "plot_all", fake_plot_all)

    bundle = api.run_main_bundle(cfg, progress=None, export_png=False)

    assert isinstance(bundle, api.MainBundle)
    assert captured["compute_cb"] is None
    assert captured["plot_cb"] is None


def test_run_main_bundle_progress_true_requires_tqdm(monkeypatch) -> None:
    cfg = api.default_config(R=10, R_garch=5, n_grid=(30,), S_grid=(0.0,), max_workers=1)
    monkeypatch.setattr(api, "_resolve_tqdm", lambda: None)

    with pytest.raises(ModuleNotFoundError):
        api.run_main_bundle(cfg, include_plots=False, progress=True, export_png=False)


def test_run_main_bundle_progress_true_requires_notebook_widget_backend(monkeypatch) -> None:
    cfg = api.default_config(R=10, R_garch=5, n_grid=(30,), S_grid=(0.0,), max_workers=1)
    monkeypatch.setattr(api, "_is_notebook_session", lambda: True)
    monkeypatch.setattr(api, "_has_ipywidgets", lambda: False)

    with pytest.raises(ModuleNotFoundError, match="ipywidgets"):
        api.run_main_bundle(cfg, include_plots=False, progress=True, export_png=False)


def test_run_main_bundle_progress_none_warns_and_disables_without_notebook_widgets(monkeypatch) -> None:
    cfg = api.default_config(R=10, R_garch=5, n_grid=(30,), S_grid=(0.0,), max_workers=1)
    results = pd.DataFrame(
        [
            {
                "dgp": "iid_normal",
                "method": api.ANALYTIC_METHOD,
                "n": 30,
                "S_true": 0.0,
                "coverage_95": 0.95,
                "reject_rate_H0_S_eq_0": 0.05,
            }
        ]
    )
    diagnostics = pd.DataFrame([{"dgp": "iid_normal", "n": 30, "S_true": 0.0, "bias": 0.0, "rmse": 0.1}])
    ci_sweep = pd.DataFrame([{"dgp": "iid_normal", "method": api.ANALYTIC_METHOD, "n": 30, "S_true": 0.0}])
    cache_meta = {"cache_hit": True, "cache_hash": "abc123"}
    captured: dict[str, object | None] = {"compute_cb": None, "plot_cb": None}

    def fake_run_ci_sweep_cached(*args, progress_callback=None, **kwargs):
        captured["compute_cb"] = progress_callback
        return results, diagnostics, ci_sweep, cache_meta

    def fake_plot_all(*args, progress_callback=None, **kwargs):
        captured["plot_cb"] = progress_callback
        return {"coverage_95": Figure()}

    monkeypatch.setattr(api, "_is_notebook_session", lambda: True)
    monkeypatch.setattr(api, "_has_ipywidgets", lambda: False)
    monkeypatch.setattr(api, "_missing_notebook_progress_warned", False)
    monkeypatch.setattr(api, "_resolve_tqdm", lambda: (_ for _ in ()).throw(AssertionError("_resolve_tqdm should not be called")))
    monkeypatch.setattr(api, "run_ci_sweep_cached", fake_run_ci_sweep_cached)
    monkeypatch.setattr(api, "plot_all", fake_plot_all)

    with pytest.warns(UserWarning, match="ipywidgets"):
        bundle = api.run_main_bundle(cfg, progress=None, export_png=False)

    assert isinstance(bundle, api.MainBundle)
    assert captured["compute_cb"] is None
    assert captured["plot_cb"] is None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        api.run_main_bundle(cfg, progress=None, export_png=False)
    assert not any("widget backend is unavailable" in str(w.message) for w in caught)


def test_run_main_bundle_exports_all_figures_as_png(tmp_path, monkeypatch) -> None:
    cfg = api.default_config(R=10, R_garch=5, n_grid=(30,), S_grid=(0.0,), max_workers=1)
    results = pd.DataFrame(
        [
            {
                "dgp": "iid_normal",
                "method": api.ANALYTIC_METHOD,
                "n": 30,
                "S_true": 0.0,
                "coverage_95": 0.95,
                "reject_rate_H0_S_eq_0": 0.05,
            }
        ]
    )
    diagnostics = pd.DataFrame([{"dgp": "iid_normal", "n": 30, "S_true": 0.0, "bias": 0.0, "rmse": 0.1}])
    ci_sweep = pd.DataFrame([{"dgp": "iid_normal", "method": api.ANALYTIC_METHOD, "n": 30, "S_true": 0.0}])
    cache_meta = {"cache_hit": True, "cache_hash": "abc123"}
    figures = {"coverage_95": Figure(), "coverage_95_vs_n": Figure(), "rmse": Figure()}

    monkeypatch.setattr(api, "run_ci_sweep_cached", lambda *args, **kwargs: (results, diagnostics, ci_sweep, cache_meta))
    monkeypatch.setattr(api, "plot_all", lambda *args, **kwargs: figures)

    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_write_image(self, path, *args, **kwargs):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        calls.append((out_path, dict(kwargs)))

    monkeypatch.setattr(BaseFigure, "write_image", fake_write_image)

    bundle = api.run_main_bundle(
        cfg,
        output_dir=tmp_path / "out",
        export_dir=tmp_path / "exports",
        export_png=True,
    )

    assert isinstance(bundle, api.MainBundle)
    assert {p.name for p, _ in calls} == {f"{name}.png" for name in figures.keys()}
    assert all(kwargs.get("width") == 3200 for _, kwargs in calls)
    assert all(kwargs.get("height") == 2000 for _, kwargs in calls)
    assert all(kwargs.get("scale") == 2.0 for _, kwargs in calls)


def test_run_oracle_bundle_raises_when_png_export_fails(tmp_path, monkeypatch) -> None:
    results = _sample_oracle_results()
    diagnostics = pd.DataFrame([{"dgp": "garch11_t", "n": 36, "S_true": 0.0, "bias": 0.0, "rmse": 0.1}])
    cache_meta = {"cache_hit": True, "cache_hash": "oracle123"}

    monkeypatch.setattr(api, "run_cached", lambda *args, **kwargs: (results, diagnostics, cache_meta))
    monkeypatch.setattr(api, "plot_oracle_coverage", lambda *args, **kwargs: Figure())
    monkeypatch.setattr(api, "plot_oracle_se", lambda *args, **kwargs: Figure())
    monkeypatch.setattr(api, "write_plotly_png", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("png failed")))

    with pytest.raises(RuntimeError, match="PNG export failed"):
        api.run_oracle_bundle(
            api.default_config(
                R=10,
                R_garch=10,
                dgps=("garch11_t",),
                methods=(api.GARCH_ORACLE_METHOD,),
                n_grid=(12, 36, 60),
                S_grid=(-0.5, 0.0, 0.5),
                max_workers=1,
            ),
            main_n_grid=(36, 60),
            appendix_n_grid=(12, 36, 60),
            export_dir=tmp_path,
        )


def test_legacy_shim_imports_warn_and_reexport() -> None:
    sys.path.insert(0, str(ROOT))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy_api = importlib.reload(importlib.import_module("src.bsc_final_api"))
        legacy_plot = importlib.reload(importlib.import_module("src.plot_oracle_coverage"))

    assert hasattr(legacy_api, "run_cached")
    assert hasattr(legacy_plot, "write_plotly_png")
    assert any("deprecated" in str(w.message).lower() for w in caught)
