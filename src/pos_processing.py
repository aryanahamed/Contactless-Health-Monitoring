import numpy as np
import scipy.signal
import cv2

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
    rgb_buffer (np.ndarray): A NumPy array of shape (N, 3) where N is the number
                             of time points (frames), and the 3 columns represent
                             the average Red, Green, and Blue values over time.

  Returns:
    np.ndarray | None: A 1D NumPy array of length N containing the raw projected
                      pulse signal (often denoted as 'S' or 'h').
                      Returns None if the buffer is too short or invalid.
  """
  if rgb_buffer is not None and rgb_buffer.shape[0] > 10:
    # Normalize RGB buffer
    mean_rgb = np.mean(rgb_buffer, axis=0)
    normalized_rgb = rgb_buffer / mean_rgb
    
    # Detrend each channel
    detrended_rgb = np.apply_along_axis(scipy.signal.detrend, 0, normalized_rgb)
    
    # Define projection plane
    P = detrended_rgb[:, 1] + detrended_rgb[:, 2]  # G + B
    Q = detrended_rgb[:, 1] - detrended_rgb[:, 2]  # G - B
    
    # Calculate alpha
    std_Q = np.std(Q)
    if std_Q != 0:
      alpha = np.std(P) / std_Q
    else:
      alpha = 0
    
    # Calculate projected signal S
    S = P + alpha * Q
    return S
  else:
    return None

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
  b, a = scipy.signal.butter(order, [low, high], btype='band')
  filtered_signal = scipy.signal.filtfilt(b, a, signal_buffer)
  return filtered_signal


def calculate_signal_quality(filtered_signal, fps):
  """
  Estimates the quality of a filtered physiological signal.
  Higher score means better quality.

  Args:
    filtered_signal (np.ndarray): The 1D filtered pulse signal (output of bandpass).
    fps (float): Frames per second, needed for frequency analysis.

  Returns:
    float: A score representing the signal quality.
  """
  if filtered_signal is not None and len(filtered_signal) > 10:
    # Calculate the power spectrum using FFT
    fft_result = np.fft.fft(filtered_signal)
    power_spectrum = np.abs(fft_result) ** 2
    freqs = np.fft.fftfreq(len(filtered_signal), 1 / fps)
    
    # Define the expected signal band
    band_min = 0.75
    band_max = 4.0
    
    # Calculate the total power within the signal band
    band_mask = (freqs >= band_min) & (freqs <= band_max)
    power_in_band = np.sum(power_spectrum[band_mask])
    
    # Calculate the total power outside the signal band
    out_band_mask = (freqs >= 0.5) & (freqs <= 10.0) & ~band_mask
    power_out_of_band = np.sum(power_spectrum[out_band_mask])
    
    # Calculate the quality score
    if power_out_of_band > 0:
      quality_score = power_in_band / power_out_of_band
    else:
      quality_score = 0
    return quality_score
  else:
    return 0

def select_best_pos_signal(region_data, fps):
  """
  Processes RGB data from multiple regions, selects the best quality
  pulse signal using POS.

  Args:
    region_data (dict): A dictionary where keys are region names (e.g., 'forehead')
                        and values are the corresponding (N, 3) RGB buffers (np.ndarray).
    fps (float): Frames per second.

  Returns:
    np.ndarray | None: The selected, filtered 1D pulse signal with the highest quality.
                       Returns None if no valid signal could be obtained.
  """
  best_signal = None
  highest_quality = -1

  for region_name, rgb_buffer in region_data.items():
    pos_signal = apply_pos_projection(rgb_buffer)
    if pos_signal is not None:
      filtered_signal = apply_butterworth_bandpass(pos_signal, 0.75, 4.0, fps)
      if filtered_signal is not None:
        quality_score = calculate_signal_quality(filtered_signal, fps)
        if quality_score > highest_quality:
          highest_quality = quality_score
          best_signal = filtered_signal
  return best_signal