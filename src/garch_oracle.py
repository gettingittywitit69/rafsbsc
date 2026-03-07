"""Compatibility shim for legacy imports."""

from __future__ import annotations

import warnings

warnings.warn(
    "src.garch_oracle is deprecated; use finance_data.bsc.garch_oracle instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .finance_data.bsc import garch_oracle as _garch_oracle

for _name in dir(_garch_oracle):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_garch_oracle, _name)

__all__ = [name for name in dir(_garch_oracle) if not name.startswith("_")]
