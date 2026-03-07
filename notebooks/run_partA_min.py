#!/usr/bin/env python3
from __future__ import annotations

import os

# clamp threads BEFORE numpy/matplotlib/pandas imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------
# minimal project-root handling
# ----------------------------
def find_project_root() -> Path:
    here = Path.cwd().resolve()
    if (here / "src").exists():
        return here
    for p in here.parents:
        if (p / "src").exists():
            return p
    raise RuntimeError("Could not find project root containing src/")

ROOT = find_project_root()
sys.path.insert(0, str(ROOT))

from src import bsc_final_runtime as rt  # noqa: E402

def run_dir() -> Path:
    path = Path(os.environ.get("EXPERIMENT_BSC_OUTPUT_DIR", "outputs/experiment_bsc"))
    return (path if path.is_absolute() else ROOT / path).resolve()

def cfg_hash(cfg: rt.Config) -> str:
    payload = {
        "cfg": asdict(cfg),
        "runtime_file": str(Path(rt.__file__).resolve()),
        "runtime_sha": hashlib.sha256(Path(rt.__file__).read_bytes()).hexdigest()[:12],
        "sharpe_mc_file": str(Path(rt.sharpe_mc.__file__).resolve()),
        "sharpe_mc_sha": hashlib.sha256(Path(rt.sharpe_mc.__file__).read_bytes()).hexdigest()[:12],
    }
    s = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]

def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")

def plot_grid(results: pd.DataFrame, cfg: rt.Config, metric: str, ylabel: str, baseline: float) -> None:
    n_vals = sorted(int(v) for v in results["n"].unique())
    fig, axes = plt.subplots(
        len(cfg.dgps),
        len(cfg.methods),
        figsize=(4.6 * len(cfg.methods), 3.2 * len(cfg.dgps)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    y_min = float(np.nanmin(results[metric].to_numpy()))
    y_max = float(np.nanmax(results[metric].to_numpy()))
    y_min = min(y_min, baseline)
    y_max = max(y_max, baseline)
    if np.isclose(y_min, y_max):
        y_min -= 0.01
        y_max += 0.01

    for i, dgp in enumerate(cfg.dgps):
        for j, method in enumerate(cfg.methods):
            ax = axes[i, j]
            sub = results[(results["dgp"] == dgp) & (results["method"] == method)]
            for n in n_vals:
                rows = sub[sub["n"] == n].sort_values("S_true")
                if rows.empty:
                    continue
                ax.plot(rows["S_true"], rows[metric], marker="o", label=f"n={n}")
            ax.axhline(baseline, linestyle=":", color="black", linewidth=1.0)
            ax.grid(alpha=0.25, linestyle=":")
            ax.set_title(f"{dgp} | {method}")
            ax.set_xlabel("True Sharpe")
            ax.set_ylabel(ylabel)
            ax.set_ylim(y_min, y_max)

    # one legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()

    out = run_dir() / "figures" / f"partA_{metric}"
    savefig(fig, out)
    plt.show()
    plt.close(fig)

def main() -> None:
    cfg = rt.config_from_env()
    out = run_dir()
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    h = cfg_hash(cfg)
    results_csv = cache / f"partA_results_{h}.csv"
    diag_csv = cache / f"partA_diagnostics_{h}.csv"

    if results_csv.exists() and diag_csv.exists():
        results = pd.read_csv(results_csv)
        diagnostics = pd.read_csv(diag_csv)
        print(f"[cache] {results_csv.name}, {diag_csv.name}")
    else:
        results, diagnostics = rt.run_partA(cfg)
        results.to_csv(results_csv, index=False)
        diagnostics.to_csv(diag_csv, index=False)
        print(f"[run] wrote {results_csv.name}, {diag_csv.name}")

    # also write "latest"
    results.to_csv(out / "results_partA.csv", index=False)
    diagnostics.to_csv(out / "diagnostics_partA.csv", index=False)

    plot_grid(results, cfg, metric="coverage_95", ylabel="95% coverage", baseline=0.95)
    plot_grid(results, cfg, metric="reject_rate_H0_S_eq_0", ylabel="Reject rate (H0: S=0)", baseline=float(cfg.alpha))

if __name__ == "__main__":
    main()
