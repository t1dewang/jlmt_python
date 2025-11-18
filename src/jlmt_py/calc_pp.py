"""
Periodicity Pitch (PP) Calculation.

This module computes the Periodicity Pitch representation from an
Auditory Nerve Image (ANI). It includes low-pass filtering,
half-wave rectification, and summed autocorrelation.

A separate function, `apply_attenuation`, is provided for
perceptual weighting.

Functions
---------
calc_pp : Calculates the raw periodicity pitch (PP) from an ANI.
apply_attenuation : Applies a perceptual attenuation window to a PP matrix.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import signal
import math
import warnings
import matplotlib.pyplot as plt


def calc_pp(
    ani,
    ani_fs,
    low_freq = 80.0,
    frame_width = 0.0381,
    frame_step = 0.0381):
    """
    Calculates the periodicity pitch from an ANI.

    This function implements the core logic from the IPEM Toolbox's
    IPEMPeriodicityPitch.m file. 

    Parameters:
        ani (np.ndarray): input ani, shape (num_channels, num_samples).
        ani_fs (float): The sample frequency of ani.
        low_freq (float): Cutoff frequency for the low-pass
            Defaults to 80.0.
        frame_width (float): The width of the analysis frame.
            Defaults to 0.0381.
        frame_step (float): The step size between frames. 
            Defaults to 0.0381.

    Returns:
        out_signal (np.ndarray): The pp matrix, shape (num_periods, num_frames).
        out_sample_freq (float): The new "sample frequency" of the output signal.
        out_periods (np.ndarray): A 1D array of the analyzed periods in seconds.
        fani (np.ndarray): The intermediate Filtered Auditory Nerve Image.
        delay (int): The calculated delay of the low-pass filter in samples.
    """

    # Filtering (Low-pass)
    num_ch, num_samples = ani.shape
    nyq = 0.5 * ani_fs    
    wn = low_freq / nyq
    if wn <= 0 or wn >= 1:
        raise ValueError(f"Invalid cutoff: {low_freq} Hz for fs={ani_fs}")
    b, a = signal.butter(2, wn, btype='low')
    
    # FANI Calculation
    lpf = signal.lfilter(b, a, ani, axis=1)
    impulse = np.zeros(1000)
    impulse[0] = 1.0
    h = signal.lfilter(b, a, impulse)
    delay = int(np.argmax(h))
    if delay >= num_samples:
        raise ValueError("Filter delay >= signal length; increase input length.")
    
    out_len = num_samples - delay # Create FANI by subtracting the delayed LPF signal from the original
    fani = ani[:, :out_len] - lpf[:, delay:delay + out_len]
    np.maximum(fani, 0.0, out=fani)  # Clip negative values to zero
    
    # Framing   
    W = int(round(frame_width * ani_fs))   # frame width in samples
    H = int(round(frame_step * ani_fs))    # frame hop in samples
    if W <= 1 or H <= 0:
        raise ValueError("Frame width/step too small.")
    W2 = 2 * W

    last_start = fani.shape[1] - W2 # Find the start index of the last possible frame
    if last_start < 0:
        raise ValueError("Signal too short for given frame width.")
    num_frames = 1 + last_start // H

    out_signal = np.zeros((W, num_frames), dtype=float)  # Initialize output matrix
    zeros_W = np.zeros(W, dtype=fani.dtype)
    out_sample_freq = ani_fs / H # Calculate output sample rate and period axis
    out_periods = np.arange(0, W) / ani_fs

    # Summed Autocorrelation Loop
    frame_idx = 0
    for i in range(0, last_start + 1, H):
        sum_ac = np.zeros(W, dtype=float)

        part1 = fani[:, i : i + W]
        part2 = fani[:, i : i + W2]

        for ch in range(num_ch):
            a_vec = np.concatenate([zeros_W, part1[ch]])
            b_vec = part2[ch]

            ac = signal.correlate(a_vec, b_vec, mode='full') # Calculate the full correlation
            center = len(ac) // 2
            ac = ac[center - W : center + W + 1]
            pos = ac[W+1 : 2*W+1]
            sum_ac += pos
        # Store the final summed autocorrelation for this frame
        out_signal[:, frame_idx] = sum_ac[::-1]
        frame_idx += 1
    return out_signal, out_sample_freq, out_periods, fani, delay


#attenuation function

def apply_attenuation(out_signal, atten_type='ipem_squash_hf'):
    """
    Applies an attenuation window along the periodicity (rows) axis.
    This corresponds to pp_atten_func.m and the final step in calc_pp.m.
    """
    
    num_periods = out_signal.shape[0]

    if atten_type is None or atten_type.lower() == 'none' or atten_type == '':
        return out_signal # No attenuation

    win = None
    if atten_type.lower() == 'ipem':
        x = np.arange(1, num_periods + 1)
        win = 1 - (((x / num_periods) - 0.5) ** 2)

    elif atten_type.lower() == 'ipem_squash_hf':
        x = np.arange(1, num_periods + 1)
        win = 1 - (((x / num_periods) - 0.5) ** 2)
        
        win[0:2] = 0.0
    
    else:
        print(f"Warning: Unknown attenuation function: {atten_type}")
        return out_signal

    win_col = win.reshape(-1, 1)

    return out_signal * win_col