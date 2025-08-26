"""Compatibility shim: import symbols from global_dela and warn about rename."""
import warnings as _warnings
from .global_dela import *  # re-export all public symbols

_warnings.warn(
    "ScanNetV2.delasemseg_attn is deprecated. Use ScanNetV2.global_dela instead.",
    DeprecationWarning,
    stacklevel=2,
)

