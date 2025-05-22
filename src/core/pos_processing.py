import numpy as np
import scipy.signal
from core.config import (
    MIN_SAMPLES_FOR_POS,
    MIN_SAMPLES_FOR_QUALITY,
    BAND_MIN_HZ,
    BAND_MAX_HZ,
    DEFAULT_TARGET_FPS
)

def handle_nan_values(rgb_data, timestamps):
    if rgb_data is None or timestamps is None or rgb_data.shape[0] != timestamps.shape[0]:
        return None, None
    nan_mask = ~np.isnan(rgb_data).any(axis=1)
    cleaned_rgb = rgb_data[nan_mask]
    cleaned_timestamps = timestamps[nan_mask]
    if cleaned_rgb.shape[0] < MIN_SAMPLES_FOR_POS:
        return None, None
    return cleaned_rgb, cleaned_timestamps

def apply_pos_projection(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0] < MIN_SAMPLES_FOR_POS:
        return None

    mean_rgb = np.mean(rgb_buffer, axis=0)
    if np.any(mean_rgb == 0):
        return None
    normalized_rgb = rgb_buffer / mean_rgb
    detrended_rgb = scipy.signal.detrend(normalized_rgb, axis=0)
    X = detrended_rgb[:, 1] - detrended_rgb[:, 2]
    Y = detrended_rgb[:, 1] + detrended_rgb[:, 2] - 2*detrended_rgb[:, 0]
    std_Y = np.std(Y)
    if std_Y == 0:
        return None
    alpha = np.std(X) / std_Y
    S = X - alpha * Y
    return S

def apply_butterworth_bandpass(signal_buffer, low_cut_hz, high_cut_hz, fps, order=5):
    if signal_buffer is None or fps <= 0:
        return None
    if len(signal_buffer) <= order * 3:
        return None
    nyquist = 0.5 * fps
    low = low_cut_hz / nyquist
    high = high_cut_hz / nyquist
    if low <= 0:
        low = 0.01
    if high >= 1:
        high = 0.99
    if low >= high:
        return None
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered_signal = scipy.signal.filtfilt(b, a, signal_buffer)
    return filtered_signal

def calculate_signal_quality(filtered_signal, fps):
    if filtered_signal is None or len(filtered_signal) < MIN_SAMPLES_FOR_QUALITY or fps <= 0:
        return 0.0
    try:
        freqs, power_spectrum = scipy.signal.welch(
            filtered_signal, fs=fps, nperseg=min(256, len(filtered_signal))
        )
        nyquist = 0.5 * fps
        signal_band_mask = (freqs >= BAND_MIN_HZ) & (freqs <= BAND_MAX_HZ)
        if not np.any(signal_band_mask):
            return 0.0
        power_in_band = np.sum(power_spectrum[signal_band_mask])
        noise_mask_low = (freqs >= 0.5) & (freqs < BAND_MIN_HZ) & (freqs < nyquist)
        noise_mask_high = (freqs > BAND_MAX_HZ) & (freqs <= 5.0) & (freqs < nyquist)
        power_noise_low = np.sum(power_spectrum[noise_mask_low])
        power_noise_high = np.sum(power_spectrum[noise_mask_high])
        noise_power = power_noise_low + power_noise_high
        epsilon = 1e-10 
        signal_to_noise_raw = power_in_band / (noise_power + epsilon)
        capped_snr = min(signal_to_noise_raw, 10.0)
        signal_variance = np.var(filtered_signal)
        variance_factor = np.clip(signal_variance * 1000, 0.0, 1.0) 
        w_snr = 0.8
        w_var = 0.2
        quality_score_unscaled = (w_snr * (capped_snr / 10.0)) + (w_var * variance_factor) 
        final_quality_score = np.clip(quality_score_unscaled * 10.0, 0.0, 10.0)
        return final_quality_score
    except Exception:
        return 0.0

def select_best_pos_signal(input_data):
    timestamps_np = np.array(input_data["timestamps"])
    region_data_with_timestamps = {}
    for region in ["forehead"]:
        if region in input_data and len(input_data[region]) == len(timestamps_np):
            rgb_array = np.array(input_data[region], dtype=float)
            region_data_with_timestamps[region] = {
                'rgb': rgb_array,
                'timestamps': timestamps_np
            }
    best_filtered_pos = None
    best_original_rgb = None
    best_original_timestamps = None
    best_fps = None
    highest_quality = -1.0
    for region, data in region_data_with_timestamps.items():
        rgb_buffer = data.get('rgb')
        timestamps = data.get('timestamps')
        if rgb_buffer is None or timestamps is None:
            continue
        if len(rgb_buffer) != len(timestamps):
            continue
        cleaned_rgb, cleaned_timestamps = handle_nan_values(rgb_buffer, timestamps)
        if cleaned_rgb is None:
            continue
        pos_signal = apply_pos_projection(cleaned_rgb)
        if pos_signal is None:
            continue
        if len(cleaned_timestamps) > 1:
            duration = cleaned_timestamps[-1] - cleaned_timestamps[0]
            fps = (len(cleaned_timestamps) - 1) / duration if duration > 0 else DEFAULT_TARGET_FPS
        else:
            fps = DEFAULT_TARGET_FPS
        filtered_pos = apply_butterworth_bandpass(pos_signal, BAND_MIN_HZ, BAND_MAX_HZ, fps)
        if filtered_pos is None:
            continue
        quality_score = calculate_signal_quality(filtered_pos, fps)
        if quality_score > highest_quality:
            highest_quality = quality_score
            best_filtered_pos = filtered_pos
            best_original_rgb = cleaned_rgb
            best_original_timestamps = cleaned_timestamps
            best_fps = fps
        
    print(f"Selected region: {region}, Quality: {highest_quality:.2f}, FPS: {fps:.2f}")
    if best_filtered_pos is not None:
        return (best_filtered_pos, best_original_rgb, best_original_timestamps, best_fps, highest_quality)
    else:
        return None, None, None, None, -1.0

