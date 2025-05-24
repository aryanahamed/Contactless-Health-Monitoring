import numpy as np
import scipy.signal
from numba import njit
from config import (
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


@njit(cache=True)
def _compute_pos_core(normalized_rgb):
    n_samples = normalized_rgb.shape[0]
    X = np.empty(n_samples, dtype=np.float32)
    Y = np.empty(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = normalized_rgb[i, 1] - normalized_rgb[i, 2]  # G - B
        Y[i] = normalized_rgb[i, 1] + normalized_rgb[i, 2] - 2 * normalized_rgb[i, 0]  # G + B - 2R
    
    std_X = np.std(X)
    std_Y = np.std(Y)
    
    if std_Y == 0:
        return None
    
    alpha = std_X / std_Y
    S = X - alpha * Y
    return S


def apply_pos_projection(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0] < MIN_SAMPLES_FOR_POS:
        return None

    mean_rgb = np.mean(rgb_buffer, axis=0)
    if np.any(mean_rgb <= 1e-10):
        return None
    
    normalized_rgb = rgb_buffer / mean_rgb
    
    scipy.signal.detrend(normalized_rgb, axis=0, overwrite_data=True)
    
    return _compute_pos_core(normalized_rgb)


_FILTER_CACHE = {}
def _get_butterworth_coeffs(low_cut_hz, high_cut_hz, fps, order=5):
    key = (low_cut_hz, high_cut_hz, fps, order)
    if key not in _FILTER_CACHE:
        nyquist = 0.5 * fps
        low = max(0.01, low_cut_hz / nyquist)
        high = min(0.99, high_cut_hz / nyquist)
        if low < high:
            _FILTER_CACHE[key] = scipy.signal.butter(order, [low, high], btype='band')
        else:
            _FILTER_CACHE[key] = None
    return _FILTER_CACHE[key]


def apply_butterworth_bandpass(signal_buffer, low_cut_hz, high_cut_hz, fps, order=5):
    if signal_buffer is None or fps <= 0 or len(signal_buffer) <= order * 3:
        return None
    
    coeffs = _get_butterworth_coeffs(low_cut_hz, high_cut_hz, fps, order)
    if coeffs is None:
        return None
    
    b, a = coeffs
    return scipy.signal.filtfilt(b, a, signal_buffer)


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


@njit(cache=True)
def _compute_power_sums(power_spectrum, freqs, band_min, band_max):
    signal_power = 0.0
    noise_power_low = 0.0
    noise_power_high = 0.0
    
    for i in range(len(freqs)):
        freq = freqs[i]
        power = power_spectrum[i]
        
        if band_min <= freq <= band_max:
            signal_power += power
        elif 0.5 <= freq < band_min:
            noise_power_low += power
        elif band_max < freq <= 5.0:
            noise_power_high += power
    
    return signal_power, noise_power_low + noise_power_high


def calculate_signal_quality(filtered_signal, fps):
    if filtered_signal is None or len(filtered_signal) < MIN_SAMPLES_FOR_QUALITY or fps <= 0:
        return 0.0
    
    try:
        nperseg = min(256, len(filtered_signal) // 4 * 4)
        if nperseg < 32:
            nperseg = min(32, len(filtered_signal))
        
        freqs, power_spectrum = scipy.signal.welch(
            filtered_signal, fs=fps, nperseg=nperseg
        )
        
        signal_power, noise_power = _compute_power_sums(
            power_spectrum, freqs, BAND_MIN_HZ, BAND_MAX_HZ
        )
        
        if signal_power == 0:
            return 0.0
        
        epsilon = 1e-10
        snr_raw = signal_power / (noise_power + epsilon)
        capped_snr = min(snr_raw, 10.0)
        
        signal_variance = np.var(filtered_signal)
        variance_factor = min(signal_variance * 1000, 1.0)
        
        quality_score = min(10.0, (0.8 * capped_snr + 0.2 * variance_factor * 10.0))
        
        return quality_score
        
    except Exception:
        return 0.0


def select_best_pos_signal(input_data):
    timestamps_np = np.asarray(input_data["timestamps"], dtype=np.float64)
    
    valid_regions = {}
    for region in ["forehead"]:
        if region in input_data and len(input_data[region]) == len(timestamps_np):
            rgb_array = np.asarray(input_data[region], dtype=np.float64)
            if rgb_array.size > 0:
                valid_regions[region] = rgb_array
    
    if not valid_regions:
        return None, None, None, None, -1.0
    
    best_result = None
    highest_quality = -1.0
    
    for region, rgb_buffer in valid_regions.items():
        if rgb_buffer.shape[0] < MIN_SAMPLES_FOR_POS:
            continue
            
        cleaned_rgb, cleaned_timestamps = handle_nan_values(rgb_buffer, timestamps_np)
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
            best_result = (filtered_pos, cleaned_rgb, cleaned_timestamps, fps, quality_score)
            
            if quality_score > 8.1:
                break
    
    return best_result if best_result else (None, None, None, None, -1.0)
