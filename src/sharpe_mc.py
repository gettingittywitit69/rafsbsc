"""Compatibility shim for legacy imports."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.sharpe_mc is deprecated; use finance_data.bsc.sharpe_mc instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .finance_data.bsc import sharpe_mc as _sharpe_mc

for _name in dir(_sharpe_mc):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_sharpe_mc, _name)

__all__ = [name for name in dir(_sharpe_mc) if not name.startswith("_")]
