"""
JLMT Python Package
===================

The JLMT package provides a complete set of audio processing utilities based on the IPEM auditory model.
It exposes a clean, modular API for signal preprocessing, auditory modeling,
feature extraction, and music-theoretic representations.

Main Features
-------------
- **load_audio**: Unified audio loading and normalization.
- **calc_ani**: Auditory Nerve Image (ANI) computation via the IPEM model.
- **calc_pp**: Phase Perception (PP) representation.
- **leaky_integrate**: Leaky integration applied to PP.
- **calc_tonal_space**: Tonal Space (TS) representation.
- **run_pipeline**: High-level, end-to-end audio processing pipeline.

Design Principles
-----------------
Each computational unit is implemented as an independent module (.py file)
to ensure clarity, maintainability, and extensibility of the codebase.
This design encourages clean architecture and makes the system suitable
for both research and production environments.

Version
-------
0.1.0

Author
------
Tao Wang

"""

from .load_audio import load_audio
from .calc_ani import calc_ani, plot_multi_channel
from .calc_pp import calc_pp, apply_attenuation
from .leaky_integrate import leaky_integrate
from .calc_tonal_space import load_som, calc_tonal_space
# from .run_pipeline import run_pipeline

# ===========================
# Low-level C auditory engine
# ===========================
# Load compiled extension module (ipem_auditory_model.pyd)
try:
    import ipem_auditory_model
except ImportError as e:
    raise ImportError(
        "The compiled extension 'ipem_auditory_model' could not be loaded.\n"
        "Make sure you have built the extension via:\n"
        "    python setup.py build_ext --inplace\n"
        "and that the resulting .pyd file is placed inside jlmt_py/"
    ) from e

__all__ = [
    "load_audio",
    "calc_ani",
    "plot_multi_channel",
    "calc_pp",
    "apply_attenuation",
    "leaky_integrate",
    "load_som",
    "calc_tonal_space",
    "run_pipeline",
    "ipem_auditory_model",
]


