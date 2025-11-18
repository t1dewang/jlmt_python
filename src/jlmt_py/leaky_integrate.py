"""
Leaky Integration Module.

This module provides a function to perform leaky integration on a
time-series representation, typically a Periodicity Pitch (PP) matrix.
This process models temporal accumulation of features and produces
a Context Image (CI).

Functions
---------
leaky_integrate : Applies leaky integration to a PP matrix.
"""

import numpy as np

def leaky_integrate(
    out_signal: np.ndarray, 
    out_sample_freq: float, 
    half_decay_time: float = 0.1, 
    enlargement: float = 0.0
) -> np.ndarray:
    """
    Performs leaky integration on a Periodicity Pitch (PP) matrix.

    This function models the temporal summation of neural activity,
    resulting in a Context Image (CI).

    Parameters
    ----------
    out_signal : np.ndarray
        Input Periodicity Pitch matrix, shape (period_bins, frames).
    out_sample_freq : float
        Sample frequency of the PP representation (Hz).
    half_decay_time : float, default=0.1
        Time (in seconds) for the integrated amplitude to decay to half.
    enlargement : float, default=0.0
        Time (in seconds) to extend the signal with zeros for decay.
        If -1, it defaults to 2 * half_decay_time.

    Returns
    -------
    np.ndarray
        The resulting leaky integrated image (Context Image), shape
        (period_bins, frames + enlarged_frames).
    """

    # Handle enlargement parameter
    if enlargement == -1:
        enlargement = 2 * half_decay_time
    enlarge_samples = int(round(out_sample_freq * enlargement))

    # Calculate the integrator coefficient based on half-decay time
    if half_decay_time > 0:
        integrator = 2 ** (-1.0 / (out_sample_freq * half_decay_time))
    else:
        integrator = 0.0

    # Create the input matrix by padding with zeros
    M = np.hstack([out_signal, np.zeros((out_signal.shape[0], enlarge_samples))])
    out_li = np.zeros_like(M)

    # Perform the integration loop
    out_li[:, 0] = M[:, 0]
    for j in range(1, M.shape[1]):
        out_li[:, j] = integrator * out_li[:, j - 1] + M[:, j]

    return out_li