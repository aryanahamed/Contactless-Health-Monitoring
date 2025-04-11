import numpy as np
import scipy.signal
import cv2

# --- Constants ---
BAND_MIN = 0.75  # 45 BPM
BAND_MAX = 3.0   # 180 BPM

def extract_average_rgb(roi_frame):
  """
  Extracts the average Red, Green, and Blue channel values from an ROI frame.

  Args:
    roi_frame (np.ndarray): The frame containing the Region of Interest
                            (expects BGR format from OpenCV).

  Returns:
    tuple[float, float, float] | None: A tuple containing (average_red, average_green, average_blue).
                                      Returns None if the frame is invalid.
  """
  if roi_frame is not None and roi_frame.ndim == 3 and roi_frame.shape[2] == 3:
    mean_channels = np.mean(roi_frame, axis=(0, 1))
    mean_rgb = (mean_channels[2], mean_channels[1], mean_channels[0])
    return mean_rgb
  else:
    return None

def apply_pos_projection(rgb_buffer):
    """
    Applies the POS algorithm projection to isolate the pulse signal.

    Args:
        rgb_buffer (np.ndarray): Shape (N, 3) for R, G, B per frame.

    Returns:
        np.ndarray | None: 1D projected pulse signal (POS 'S').
    """
    if rgb_buffer is None or rgb_buffer.shape[0] < 30:
        return None

    # Normalize by mean per channel
    mean_rgb = np.mean(rgb_buffer, axis=0)
    normalized_rgb = rgb_buffer / mean_rgb

    # Detrend
    detrended_rgb = scipy.signal.detrend(normalized_rgb, axis=0)

    # Apply projection matrix
    X = detrended_rgb[:, 1] - detrended_rgb[:, 2] # G - B
    Y = detrended_rgb[:, 1] + detrended_rgb[:, 2] - 2*detrended_rgb[:, 0]  # G + B - 2R

    std_Y = np.std(Y)
    if std_Y == 0:
        return None

    alpha = np.std(X) / std_Y
    S = X - alpha * Y

    return S

def apply_butterworth_bandpass(signal_buffer, low_cut, high_cut, fps, order=5):
  """
  Applies a Butterworth bandpass filter to a time-series signal buffer.

  Args:
    signal_buffer (np.ndarray): A 1D buffer containing the signal values
                                over time (e.g., the output from POS projection).
    low_cut (float): The lower cutoff frequency (in Hz).
    high_cut (float): The upper cutoff frequency (in Hz).
    fps (float): The frame rate of the video (samples per second).
    order (int): The order of the Butterworth filter.

  Returns:
    np.ndarray: The filtered signal.
  """
  nyquist = 0.5 * fps
  low = low_cut / nyquist
  high = high_cut / nyquist
  if high >= 1:
    high = 0.99  # Avoid instability
  b, a = scipy.signal.butter(order, [low, high], btype='band')
  filtered_signal = scipy.signal.filtfilt(b, a, signal_buffer)
  return filtered_signal


def calculate_signal_quality(filtered_signal, fps):
    """
    Estimates the quality of a filtered physiological signal.
    Higher score means better quality. This version avoids peak detection
    and relies on power in the heart rate band and signal variance.

    Args:
        filtered_signal (np.ndarray): The 1D filtered pulse signal (output of bandpass).
        fps (float): Frames per second, needed for frequency analysis.

    Returns:
        float: A score from 0.0 to 10.0 representing the signal quality.
    """
    if filtered_signal is None or len(filtered_signal) < 10:
        return 0.0

    # Power Spectral Density
    fft_result = np.fft.fft(filtered_signal)
    power_spectrum = np.abs(fft_result) ** 2
    freqs = np.fft.fftfreq(len(filtered_signal), 1 / fps)

    # Power in the desired heart rate band
    band_mask = (freqs >= BAND_MIN) & (freqs <= BAND_MAX)
    power_in_band = np.sum(power_spectrum[band_mask])

    # Total power in a broader reasonable band
    total_mask = (freqs >= 0.5) & (freqs <= 10.0)
    total_power = np.sum(power_spectrum[total_mask]) + 1e-10

    signal_to_noise = power_in_band / (total_power - power_in_band + 1e-10)

    # Signal Variance
    signal_variance = np.var(filtered_signal)
    variance_factor = min(signal_variance * 1000, 1.0)

    # Combine Factors into Quality Score
    w_snr = 0.8
    w_var = 0.2
    quality_score = (w_snr * signal_to_noise) + (w_var * variance_factor)

    # Scale to 0-10
    return min(max(quality_score * 2.0, 0.0), 10.0)


def select_best_pos_signal(region_data, fps):
  """
  Processes RGB data from multiple regions, selects the best quality
  pulse signal using POS. Also stores the RGB buffer that produced it.

  Args:
    region_data (dict): A dictionary where keys are region names (e.g., 'forehead')
                        and values are the corresponding (N, 3) RGB buffers (np.ndarray).
    fps (float): Frames per second.

  Returns:
    tuple(np.ndarray, np.ndarray, float) | tuple(None, None, float):
        - The selected, filtered 1D pulse signal (for HR/HRV).
        - The corresponding RGB buffer that generated this signal (for BR).
        - The quality score of the selected HR signal.
       Returns (None, None, -1) if no valid signal could be obtained.
  """
  best_signal = None
  best_rgb_buffer = None
  highest_quality = -1

  for region_name, rgb_buffer in region_data.items():
    # Ensure buffer is long enough for POS and BR calculations
    if rgb_buffer is None or len(rgb_buffer) < 60:
        print(f"Skipping region {region_name} due to insufficient data length: {len(rgb_buffer) if rgb_buffer is not None else 0}")
        continue
    try:
      pos_signal = apply_pos_projection(rgb_buffer)
      
      if pos_signal is not None:
        filtered_signal = apply_butterworth_bandpass(pos_signal, BAND_MIN, BAND_MAX, fps)
        
        if filtered_signal is not None:
          quality_score = calculate_signal_quality(filtered_signal, fps)
          # filtered_signal = lowpass_filter(filtered_signal, cutoff_hz=3.0, fs=fps)

          
          if quality_score > highest_quality:
            highest_quality = quality_score
            best_signal = filtered_signal
            best_rgb_buffer = rgb_buffer
                
    except Exception as e:
      print(f"Error processing region {region_name}: {e}")
      continue
  
  if best_signal is not None:
      return best_signal, best_rgb_buffer, highest_quality
  else:
      return None, None, -1
    
# def lowpass_filter(signal, cutoff_hz, fs, order=2):
#     b, a = scipy.signal.butter(order, cutoff_hz / (0.5 * fs), btype='low')
#     return scipy.signal.filtfilt(b, a, signal)