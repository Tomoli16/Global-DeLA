"""
Compatibility shim: module renamed to `global_dela`.
This file remains to avoid breaking existing imports. Please update your imports to:

    from global_dela import DelaSemSeg

"""
import warnings

warnings.warn(
    "Module 'delasemseg_attn' has been renamed to 'global_dela'. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from global_dela import *  # re-export all public symbols

