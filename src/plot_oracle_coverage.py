"""Compatibility shim for legacy imports."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.plot_oracle_coverage is deprecated; use finance_data.bsc.plotting instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .finance_data.bsc import plotting as _plotting

for _name in dir(_plotting):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_plotting, _name)

__all__ = [name for name in dir(_plotting) if not name.startswith("_")]
