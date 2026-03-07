from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np

from . import api as bsc_api
from . import runtime as bsc_runtime
from .runtime import ANALYTIC_METHOD, GARCH_MLE_METHOD, GARCH_ORACLE_METHOD, Config


@dataclass(frozen=True)
class RuntimeEstimate:
    eta_seconds: float
    eta_low_seconds: float
    eta_high_seconds: float
    cache_hit: bool
    pilot_cells: int
    pilot_R: int
    pilot_R_garch: int
    notes: str


def _build_specs(cfg: Config) -> list[tuple[str, int, float]]:
    return [(str(d), int(n), float(s)) for d, n, s in product(cfg.dgps, cfg.n_grid, cfg.S_grid)]


def _work_units_for_cell(
    *,
    dgp: str,
    methods: Sequence[str],
    R: int,
    R_garch: int,
) -> float:
    units = 0.0
    if ANALYTIC_METHOD in methods:
        units += float(R)
    if GARCH_MLE_METHOD in methods:
        units += float(R_garch)
    if GARCH_ORACLE_METHOD in methods and dgp == "garch11_t":
        units += float(R)
    return units


def _total_work_units(cfg: Config, specs: Sequence[tuple[str, int, float]]) -> float:
    return float(
        sum(
            _work_units_for_cell(
                dgp=dgp,
                methods=cfg.methods,
                R=int(cfg.R),
                R_garch=int(cfg.R_garch),
            )
            for dgp, _, _ in specs
        )
    )


def _cache_hit_for_main(
    cfg: Config,
    *,
    ci_levels: tuple[float, ...],
    scope: str,
    output_dir: str | Path | None,
) -> bool:
    cache_scope = bsc_api._normalize_scope(scope)
    run_dir = bsc_api._resolve_run_dir(output_dir)
    cache_dir = run_dir / "cache"
    cache_hash = bsc_api._cache_hash(cfg, cache_scope, ci_levels=ci_levels)

    results_base = cache_dir / f"{cache_scope}_results_{cache_hash}"
    diagnostics_base = cache_dir / f"{cache_scope}_diagnostics_{cache_hash}"
    ci_sweep_base = cache_dir / f"{cache_scope}_ci_sweep_{cache_hash}"
    return (
        bsc_api._has_cache_artifact(results_base)
        and bsc_api._has_cache_artifact(diagnostics_base)
        and bsc_api._has_cache_artifact(ci_sweep_base)
    )


def _select_pilot_specs(
    specs: Sequence[tuple[str, int, float]],
    *,
    pilot_cells: int,
) -> list[tuple[str, int, float]]:
    if not specs:
        return []
    sorted_specs = sorted(specs, key=lambda x: (x[0], x[1], x[2]))
    if pilot_cells >= len(sorted_specs):
        return list(sorted_specs)

    selected: list[tuple[str, int, float]] = []
    seen: set[tuple[str, int, float]] = set()

    by_dgp: dict[str, list[tuple[str, int, float]]] = {}
    for spec in sorted_specs:
        by_dgp.setdefault(spec[0], []).append(spec)

    for dgp in sorted(by_dgp):
        bucket = by_dgp[dgp]
        mid = bucket[len(bucket) // 2]
        if mid not in seen:
            selected.append(mid)
            seen.add(mid)
        if len(selected) >= pilot_cells:
            return selected[:pilot_cells]

    if len(selected) < pilot_cells:
        idxs = np.linspace(0, len(sorted_specs) - 1, num=pilot_cells, dtype=int).tolist()
        for idx in idxs:
            spec = sorted_specs[int(idx)]
            if spec in seen:
                continue
            selected.append(spec)
            seen.add(spec)
            if len(selected) >= pilot_cells:
                break

    return selected[:pilot_cells]


def _resolve_pilot_reps(
    cfg: Config,
    *,
    pilot_R: int | None,
    pilot_R_garch: int | None,
) -> tuple[int, int]:
    default_R = int(min(max(20, int(cfg.R) // 20), max(20, int(cfg.R))))
    default_Rg = int(min(max(10, int(cfg.R_garch) // 20), max(10, int(cfg.R_garch))))

    R = int(default_R if pilot_R is None else pilot_R)
    Rg = int(default_Rg if pilot_R_garch is None else pilot_R_garch)
    R = max(1, min(int(cfg.R), R))
    Rg = max(1, min(int(cfg.R_garch), Rg))
    return R, Rg


def _time_single_cell(
    spec: tuple[str, int, float],
    *,
    cfg: Config,
    ci_levels: tuple[float, ...],
) -> float:
    dgp, n, s_true = spec
    start = time.perf_counter()
    bsc_runtime._run_cell_impl(dgp, int(n), float(s_true), cfg, ci_levels=ci_levels)
    return float(time.perf_counter() - start)


def _time_pilot_specs_parallel(
    specs: Sequence[tuple[str, int, float]],
    *,
    cfg: Config,
    ci_levels: tuple[float, ...],
    workers: int,
) -> float:
    if not specs:
        return 0.0
    worker_count = max(1, int(workers))
    payload = [(str(d), int(n), float(s), cfg, ci_levels) for d, n, s in specs]
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as ex:
        for _ in ex.map(bsc_runtime._run_cell_with_ci_sweep, payload):
            pass
    return float(time.perf_counter() - start)


def estimate_main_bundle_runtime(
    cfg: Config,
    *,
    ci_levels: tuple[float, ...] | list[float] = (0.90, 0.95, 0.975, 0.99),
    scope: str = "main",
    output_dir: str | Path | None = None,
    pilot_cells: int = 4,
    pilot_R: int | None = None,
    pilot_R_garch: int | None = None,
) -> RuntimeEstimate:
    normalized_levels = bsc_runtime.normalize_ci_levels(ci_levels)
    pilot_cells = max(1, int(pilot_cells))
    specs = _build_specs(cfg)

    if _cache_hit_for_main(cfg, ci_levels=normalized_levels, scope=scope, output_dir=output_dir):
        return RuntimeEstimate(
            eta_seconds=0.0,
            eta_low_seconds=0.0,
            eta_high_seconds=0.0,
            cache_hit=True,
            pilot_cells=0,
            pilot_R=0,
            pilot_R_garch=0,
            notes="Cache artifacts already exist for this config and ci_levels.",
        )

    selected_specs = _select_pilot_specs(specs, pilot_cells=pilot_cells)
    selected_count = len(selected_specs)
    resolved_R, resolved_Rg = _resolve_pilot_reps(cfg, pilot_R=pilot_R, pilot_R_garch=pilot_R_garch)
    pilot_cfg = replace(cfg, R=int(resolved_R), R_garch=int(resolved_Rg))

    rates: list[float] = []
    serial_probe_seconds = 0.0
    for spec in selected_specs:
        elapsed = _time_single_cell(spec, cfg=pilot_cfg, ci_levels=normalized_levels)
        units = _work_units_for_cell(
            dgp=spec[0],
            methods=cfg.methods,
            R=resolved_R,
            R_garch=resolved_Rg,
        )
        if units > 0.0 and np.isfinite(elapsed):
            rates.append(float(elapsed / units))
            serial_probe_seconds += float(elapsed)

    if not rates:
        return RuntimeEstimate(
            eta_seconds=0.0,
            eta_low_seconds=0.0,
            eta_high_seconds=0.0,
            cache_hit=False,
            pilot_cells=selected_count,
            pilot_R=resolved_R,
            pilot_R_garch=resolved_Rg,
            notes="No valid pilot rate observations were produced.",
        )

    target_units = _total_work_units(cfg, specs)
    q20, q50, q80 = np.quantile(np.asarray(rates, dtype=float), [0.2, 0.5, 0.8]).tolist()

    workers = max(1, min(int(cfg.max_workers), max(1, len(specs))))
    observed_speedup = 1.0
    speedup_note = "observed_speedup=1.00 (single-worker run)."
    if workers > 1:
        if selected_count < 2 or (not np.isfinite(serial_probe_seconds)) or serial_probe_seconds <= 0.0:
            speedup_note = "insufficient pilot data for parallel probe; used conservative speedup=1.00."
        else:
            probe_workers = max(1, min(workers, selected_count))
            try:
                probe_parallel_seconds = _time_pilot_specs_parallel(
                    selected_specs,
                    cfg=pilot_cfg,
                    ci_levels=normalized_levels,
                    workers=probe_workers,
                )
                if np.isfinite(probe_parallel_seconds) and probe_parallel_seconds > 0.0:
                    raw_speedup = serial_probe_seconds / float(probe_parallel_seconds)
                    observed_speedup = float(np.clip(raw_speedup, 1.0, float(workers)))
                    speedup_note = (
                        f"observed_speedup={observed_speedup:.2f} "
                        f"(probe_serial={serial_probe_seconds:.2f}s, "
                        f"probe_parallel={float(probe_parallel_seconds):.2f}s, "
                        f"probe_workers={probe_workers})."
                    )
                else:
                    speedup_note = "parallel probe produced invalid elapsed; used conservative speedup=1.00."
            except Exception as exc:
                speedup_note = f"parallel probe failed ({exc.__class__.__name__}); used conservative speedup=1.00."
    parallel_scale = 1.0 / observed_speedup

    eta = max(float(q50 * target_units * parallel_scale), 0.0)
    eta_low = max(float(q20 * target_units * parallel_scale), 0.0)
    eta_high = max(float(q80 * target_units * parallel_scale), 0.0)
    eta_low = min(eta_low, eta, eta_high)
    eta_high = max(eta_low, eta, eta_high)

    note = (
        f"Pilot sampled {selected_count}/{len(specs)} cells with R={resolved_R}, "
        f"R_garch={resolved_Rg}; workers adjustment uses max_workers={workers}, "
        f"{speedup_note}"
    )
    return RuntimeEstimate(
        eta_seconds=eta,
        eta_low_seconds=eta_low,
        eta_high_seconds=eta_high,
        cache_hit=False,
        pilot_cells=selected_count,
        pilot_R=resolved_R,
        pilot_R_garch=resolved_Rg,
        notes=note,
    )


def _parse_csv_values(raw: str | None, cast):
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        return None
    return tuple(cast(part) for part in parts)


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = bsc_runtime.config_from_env()
    payload: dict[str, object] = {}

    if args.R is not None:
        payload["R"] = int(args.R)
    if args.R_garch is not None:
        payload["R_garch"] = int(args.R_garch)
    if args.max_workers is not None:
        payload["max_workers"] = int(args.max_workers)

    dgps = _parse_csv_values(args.dgps, str)
    if dgps is not None:
        payload["dgps"] = dgps

    methods = _parse_csv_values(args.methods, str)
    if methods is not None:
        payload["methods"] = methods

    n_grid = _parse_csv_values(args.n_grid, int)
    if n_grid is not None:
        payload["n_grid"] = n_grid

    s_grid = _parse_csv_values(args.s_grid, float)
    if s_grid is not None:
        payload["S_grid"] = s_grid

    if not payload:
        return cfg
    return replace(cfg, **payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate runtime for finance_data.bsc.run_main_bundle")
    parser.add_argument("--scope", default="main", help="Cache scope used by run_main_bundle")
    parser.add_argument("--output-dir", default=None, help="Optional output directory used by run_main_bundle")
    parser.add_argument("--ci-levels", default="0.90,0.95,0.975,0.99", help="Comma-separated CI levels")
    parser.add_argument("--pilot-cells", type=int, default=4, help="Number of pilot cells to sample")
    parser.add_argument("--pilot-R", type=int, default=None, help="Pilot R override")
    parser.add_argument("--pilot-R-garch", type=int, default=None, help="Pilot R_garch override")

    parser.add_argument("--R", type=int, default=None, help="Override total outer reps")
    parser.add_argument("--R-garch", dest="R_garch", type=int, default=None, help="Override GARCH outer reps")
    parser.add_argument("--max-workers", type=int, default=None, help="Override max_workers")
    parser.add_argument("--dgps", default=None, help="Override dgps as comma-separated list")
    parser.add_argument("--methods", default=None, help="Override methods as comma-separated list")
    parser.add_argument("--n-grid", default=None, help="Override n_grid as comma-separated ints")
    parser.add_argument("--s-grid", default=None, help="Override S_grid as comma-separated floats")
    return parser


def _fmt_duration(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    if seconds < 120.0:
        return f"{seconds:.1f}s"
    return f"{seconds / 60.0:.1f}m"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    ci_levels = _parse_csv_values(args.ci_levels, float) or (0.90, 0.95, 0.975, 0.99)
    cfg = _config_from_args(args)

    estimate = estimate_main_bundle_runtime(
        cfg,
        ci_levels=ci_levels,
        scope=args.scope,
        output_dir=args.output_dir,
        pilot_cells=args.pilot_cells,
        pilot_R=args.pilot_R,
        pilot_R_garch=args.pilot_R_garch,
    )

    print(
        f"ETA: {_fmt_duration(estimate.eta_seconds)} "
        f"(range {_fmt_duration(estimate.eta_low_seconds)} - {_fmt_duration(estimate.eta_high_seconds)})"
    )
    print(f"cache_hit={estimate.cache_hit}")
    print(
        f"pilot_cells={estimate.pilot_cells} pilot_R={estimate.pilot_R} pilot_R_garch={estimate.pilot_R_garch}"
    )
    print(f"notes={estimate.notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
