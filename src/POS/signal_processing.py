import numpy as np
import scipy.signal
from POS.chrom_processing import apply_chrom_projection
from POS.pbv_processing import apply_pbv_projection
from POS.pos_processing import apply_pos_projection
from config import (
    MIN_SAMPLES_FOR_POS,
    MIN_SAMPLES_FOR_QUALITY,
    BAND_MIN_HZ,
    BAND_MAX_HZ,
    DEFAULT_TARGET_FPS
)
# This is where we handle the entire signal processing pipeline for all algos

def apply_windowing(signal, window_type='hann'):
    if signal is None or len(signal) == 0:
        return None
    
    if window_type == 'hann':
        window = scipy.signal.windows.hann(len(signal))
    elif window_type == 'hamming':
        window = scipy.signal.windows.hamming(len(signal))
    else:
        return signal
    
    return signal * window


def handle_nan_values(rgb_data, timestamps):
    if rgb_data is None or timestamps is None or rgb_data.shape[0] != timestamps.shape[0]:
        return None, None
    nan_mask = ~np.isnan(rgb_data).any(axis=1)
    cleaned_rgb = rgb_data[nan_mask]
    cleaned_timestamps = timestamps[nan_mask]
    if cleaned_rgb.shape[0] < MIN_SAMPLES_FOR_POS:
        return None, None
    return cleaned_rgb, cleaned_timestamps


_FILTER_CACHE = {}
def _get_butterworth_coeffs(low_cut_hz, high_cut_hz, fps, order=3):
    # Cache filter coefficients
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


def apply_butterworth_bandpass(signal_buffer, low_cut_hz, high_cut_hz, fps, order=3):
    if signal_buffer is None or fps <= 0 or len(signal_buffer) <= order * 3:
        return None
    
    coeffs = _get_butterworth_coeffs(low_cut_hz, high_cut_hz, fps, order)
    if coeffs is None:
        return None
    
    b, a = coeffs
    return scipy.signal.filtfilt(b, a, signal_buffer)



def calculate_signal_quality(filtered_signal, fps):
    # SNR Calculation for signal quality
    if filtered_signal is None or len(filtered_signal) < MIN_SAMPLES_FOR_QUALITY or fps <= 0:
        return 0.0

    try:
        nperseg = min(len(filtered_signal), 256)
        nfft = 4096
        freqs, psd = scipy.signal.welch(filtered_signal, fs=fps, nperseg=nperseg, nfft=nfft)

        hr_band_mask = (freqs >= BAND_MIN_HZ) & (freqs <= BAND_MAX_HZ)
        if not np.any(hr_band_mask):
            return 0.0
            
        freqs_hr = freqs[hr_band_mask]
        psd_hr = psd[hr_band_mask]

        if np.sum(psd_hr) < 1e-10:
            return 0.0

        peak_idx = np.argmax(psd_hr)
        f_peak = freqs_hr[peak_idx]

        signal_window_hz = 0.2
        signal_mask = (freqs_hr >= f_peak - signal_window_hz) & (freqs_hr <= f_peak + signal_window_hz)
        
        signal_power = np.sum(psd_hr[signal_mask])

        total_power_in_band = np.sum(psd_hr)
        noise_power = total_power_in_band - signal_power

        epsilon = 1e-10
        snr = signal_power / (noise_power + epsilon)
        
        quality_score = np.log1p(snr) * 2.0
        
        return min(quality_score, 10.0)

    except Exception:
        return 0.0

def select_best_signal(input_data):
    # Chooses best signal among POS, CHROM, and PBV(based on quality score)
    timestamps_np = np.asarray(input_data["timestamps"], dtype=np.float64)
    
    valid_regions = {}
    # Primarily using forehead because always better accuracy than cheeks.
    for region in ["forehead"]:
        if region in input_data and len(input_data[region]) == len(timestamps_np):
            rgb_array = np.asarray(input_data[region], dtype=np.float64)
            if rgb_array.size > 0:
                valid_regions[region] = rgb_array
    
    if not valid_regions:
        return None, None, None, None, -1.0, "none"
    
    best_result = None
    highest_quality = -1.0
    best_method = "none"
    
    for region, rgb_buffer in valid_regions.items():
        if rgb_buffer.shape[0] < MIN_SAMPLES_FOR_POS:
            continue
            
        cleaned_rgb, cleaned_timestamps = handle_nan_values(rgb_buffer, timestamps_np)
        if cleaned_rgb is None:
            continue
        
        if len(cleaned_timestamps) > 1:
            duration = cleaned_timestamps[-1] - cleaned_timestamps[0]
            fps = (len(cleaned_timestamps) - 1) / duration if duration > 0 else DEFAULT_TARGET_FPS
        else:
            fps = DEFAULT_TARGET_FPS
        
        for method_name, projection_func in [
            ("POS", apply_pos_projection),
            ("CHROM", apply_chrom_projection),
            ("PBV", apply_pbv_projection)
        ]:
            signal = projection_func(cleaned_rgb)
            
            if signal is None:
                continue
            
            # Need pre windowed for plot
            pre_windowed = apply_butterworth_bandpass(signal, BAND_MIN_HZ, BAND_MAX_HZ, fps, order=3)
            if pre_windowed is None:
                continue

            filtered_signal = apply_windowing(pre_windowed, window_type='hann')
            quality_score = calculate_signal_quality(filtered_signal, fps)
            
            if quality_score > highest_quality:
                highest_quality = quality_score
                best_result = (filtered_signal, pre_windowed, cleaned_rgb, cleaned_timestamps, fps, quality_score)
                best_method = f"{region}_{method_name}"
                
                # Early exit if quality is very high
                if quality_score > 8.5:
                    break
        
        # Early exit if we already have a very high quality signal
        if highest_quality > 8.5:
            break
        
    # Unpacks best result values and returns with best method
    return (*best_result, best_method) if best_result else (None, None, None, None, None, -1.0, "none")
