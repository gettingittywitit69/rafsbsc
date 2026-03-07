"""Compatibility shim for legacy imports."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.bsc_final_runtime is deprecated; use finance_data.bsc.runtime instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .finance_data.bsc import runtime as _runtime

for _name in dir(_runtime):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_runtime, _name)

__all__ = [name for name in dir(_runtime) if not name.startswith("_")]
