"""
load_audio.py
=============

Utility for loading and preprocessing audio files.

This module provides a unified interface for reading audio signals from disk,
normalizing them, converting to mono, and optionally resampling. It serves as
the first stage of the JLMT auditory processing pipeline.

Functions
---------
load_audio : Load an audio file with optional mono conversion, normalization,
             and resampling.

"""

from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
import librosa


def load_audio(
    path: str,
    mono: bool = True,
    normalize: bool = True,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess an audio file.

    Parameters
    ----------
    path : str
        Path to the audio file (wav/mp3/flac/ogg/...).
    mono : bool, default=True
        If True, the audio is converted to mono.
    normalize : bool, default=True
        If True, scale the waveform to the range [-1, 1].
    target_sr : int or None, default=None
        If provided, the audio will be resampled to this sampling rate.
        If None, the original sampling rate is preserved.

    Returns
    -------
    y : np.ndarray
        Audio signal as a float32 numpy array.
    sr : int
        Sampling rate of the returned signal.

    Notes
    -----
    - This function uses librosa.load() for robust multi-format audio reading.
    - Normalization ensures stable input to subsequent auditory models.
    - Mono conversion simplifies processing in the JLMT pipeline.

    Examples
    --------
    >>> y, sr = load_audio("audio.wav")
    >>> y.shape
    (48000,)
    """

    if not isinstance(path, str):
        raise TypeError("`path` must be a string representing the file path.")

    try:
        # librosa automatically handles multi-format decoding
        y, sr = librosa.load(path, sr=target_sr, mono=mono)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file '{path}': {e}")

    if normalize:
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

    return y.astype(np.float32), sr
