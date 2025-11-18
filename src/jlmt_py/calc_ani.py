"""
calc_ani.py
===========

ANI computation and visualization module for JLMT.

This module includes:
1. calc_ani() — computes the Auditory Nerve Image using the compiled
   ipem_auditory_model C extension (via process_file), following exactly
   the pipeline logic used in your original Jupyter notebook.
2. plot_multi_channel() — visualizes the multi-channel output.
"""

import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

# Load compiled C extension
try:
    import ipem_auditory_model
except ImportError:
    raise ImportError(
        "Failed to import 'ipem_auditory_model'. "
        "Ensure that the .pyd file exists inside jlmt_py/."
    )


# ============================================================
#                   High-level ANI Wrapper
# ============================================================
def calc_ani(
    in_signal: np.ndarray,
    in_sample_freq: int,
    in_auditory_model_path: str = "./",
    downsampling_factor: int = 4,
    num_channels: int = 40,
    first_cbu: float = 2.0,
    cbu_step: float = 0.5,
):
    """
    Compute the Auditory Nerve Image (ANI) using the IPEM auditory model.

    This version faithfully reproduces the logic from the original Jupyter
    notebook pipeline:

    1. Resample to 22,050 Hz
    2. Apply 20 ms silence padding
    3. Export a temporary WAV
    4. Call ipem_auditory_model.process_file() from the .pyd module
    5. Read nerve_image.ani (text data)
    6. Read filter frequencies
    7. Remove padding and downsample

    Parameters
    ----------
    in_signal : np.ndarray
        Mono audio waveform.
    in_sample_freq : int
        Sampling rate.
    in_auditory_model_path : str
        Directory for saving intermediate files.
    downsampling_factor : int
        Sampling rate reduction factor.
    num_channels : int
        Number of auditory filterbank channels.
    first_cbu : float
        Initial Channel Bandwidth Unit.
    cbu_step : float
        Step between CBUs.

    Returns
    -------
    out_ani : np.ndarray (num_channels × frames)
    out_ani_freq : float
        Sampling rate for ANI
    out_filter_freqs : np.ndarray
        Filter center frequencies (Hz)
    """

    print("Start of calc_ani...")

    # Ensure mono
    if in_signal.ndim > 1:
        if in_signal.shape[1] == 1:
            in_signal = in_signal[:, 0]
        else:
            raise ValueError("Input signal must be mono.")

    # Create output directory
    os.makedirs(in_auditory_model_path, exist_ok=True)

    # ======================================
    # 1. Resample → 22,050 Hz
    # ======================================
    target_fs = 22050
    pad_n = int(round(0.020 * target_fs))  # 20 ms
    pad = np.zeros(pad_n)

    if in_sample_freq != target_fs:
        new_signal = signal.resample_poly(in_signal, target_fs, in_sample_freq)
    else:
        new_signal = in_signal

    # Add silence padding
    padded_signal = np.concatenate([pad, new_signal, pad])

    # Save temporary WAV file
    input_wav = os.path.join(in_auditory_model_path, "input.wav")
    sf.write(input_wav, padded_signal, target_fs, subtype="PCM_16")

    # ======================================
    # 2. Call C extension process_file()
    # ======================================
    ani_file = os.path.join(in_auditory_model_path, "nerve_image.ani")

    try:
        ipem_auditory_model.process_file(
            "input.wav",
            "nerve_image.ani",
            num_channels=num_channels,
            first_freq=first_cbu,
            freq_dist=cbu_step,
            input_filepath=in_auditory_model_path,
            output_filepath=in_auditory_model_path,
            sample_frequency=target_fs,
            sound_format="wav"
        )
    except Exception as e:
        raise RuntimeError(f"Error in ipem_auditory_model.process_file(): {e}")

    # ======================================
    # 3. Load ANI output
    # ======================================
    ani_flat = np.fromfile(ani_file, sep=" ")
    out_ani = ani_flat.reshape((num_channels, -1), order="F")

    # ======================================
    # 4. Load filter frequencies (kHz → Hz)
    # ======================================
    filt_file = os.path.join(in_auditory_model_path, "FilterFrequencies.txt")
    if os.path.exists(filt_file):
        out_filter_freqs = np.loadtxt(filt_file) * 1000.0
    else:
        out_filter_freqs = np.linspace(200, 8000, num_channels)

    # ======================================
    # 5. Remove padding
    # ======================================
    trim = pad_n // 2
    out_ani = out_ani[:, trim:-trim]

    # ======================================
    # 6. Downsample ANI
    # ======================================
    out_ani_freq = target_fs / 2  # after envelope extraction
    if downsampling_factor != 1:
        out_ani = signal.resample_poly(out_ani.T, 1, downsampling_factor).T
        out_ani_freq /= downsampling_factor

    # ======================================
    # Cleanup temporary files
    # ======================================
    temp_files = [
        "input.wav", "nerve_image.ani", "FilterFrequencies.txt",
        "decim.dat", "eef.dat", "filters.dat", "lpf.dat",
        "omef.dat", "outfile.dat"
    ]
    for tf in temp_files:
        fp = os.path.join(in_auditory_model_path, tf)
        if os.path.exists(fp):
            os.remove(fp)

    print("...end of calc_ani.")

    return out_ani, out_ani_freq, out_filter_freqs


# ============================================================
#                Visualization Utility
# ============================================================
#Ploting
def plot_multi_channel(
   data,
   sample_freq=1,
   title='',
   xlabel='',
   ylabel='',
   font_size=None,
   channel_labels=None,
   channel_label_step=1,
   min_y=None,
   max_y=None,
   plot_type=0,
   time_offset=0,
   y_scale_factor=1,
   channels=None,
   one_based_index=True
):
  
   if data is None or len(data) == 0:
       print("None")
       return
  
   n_rows, n_cols = data.shape


   if channel_labels is None:
       channel_labels = [i + 1 for i in range(n_rows)]


   if channels is not None:
       if one_based_index:
           channels = [ch - 1 for ch in channels]
       channels = [ch for ch in channels if 0 <= ch < n_rows]
       if not channels:
           raise ValueError("No valid channel index")


       data = data[channels, :]
       channel_labels = [channel_labels[i] for i in channels]
       n_rows = len(channels)


   if min_y is None:
       min_y = np.min(data)
   if max_y is None:
       max_y = np.max(data)


   time_scale = np.arange(n_cols) / sample_freq + time_offset
   scale = abs(max_y - min_y) / y_scale_factor if y_scale_factor != 0 else 1


   plt.figure(figsize=(10, 6))


   if plot_type in [0, 2]:
       height = 0.9
       offset = height / 2 if plot_type == 2 else 0


       for i in range(n_rows):
           y_vals = i + height * (data[i, :] - min_y) / scale - offset
           plt.plot(time_scale, y_vals, linewidth=0.8)
      
       plt.ylim(1 - offset - 0.1, n_rows + height - offset + 0.1)
       plt.xlim(time_scale[0], time_scale[-1])


   elif plot_type == 1:
       plt.imshow(data, aspect='auto', origin='lower',
                  extent=[time_scale[0], time_scale[-1], 1, n_rows],
                  cmap='gray_r')
   else:
       raise ValueError("Unsupported plot_type (must be 0, 1, or 2)")


   if channel_label_step == -1:
       plt.yticks(np.arange(1, n_rows + 1), channel_labels)
   else:
       ticks = np.arange(0, n_rows, channel_label_step)
       plt.yticks(ticks + 1, [channel_labels[i] for i in ticks])


   plt.xlabel(xlabel)
   plt.ylabel(ylabel)
   plt.title(title)


   if font_size:
       plt.xticks(fontsize=font_size - 2)
       plt.yticks(fontsize=font_size - 2)
       plt.title(title, fontsize=font_size)


   plt.tight_layout()
   plt.show()



__all__ = ["calc_ani", "plot_multi_channel"]
