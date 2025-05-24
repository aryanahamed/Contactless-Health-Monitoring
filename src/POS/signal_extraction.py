import numpy as np
import scipy.signal
import scipy.stats
from config import (
    MAX_HR_HZ,
    MIN_PEAKS_FOR_HRV,
    MIN_VALID_IBI_S,
    MAX_VALID_IBI_S,
    MAX_ACCEPTABLE_SDNN_MS,
    MAX_ACCEPTABLE_RMSSD_MS,
    MIN_BR_HZ,
    MAX_BR_HZ,
    MIN_SAMPLES_FOR_BR,
)

from POS.pos_processing import apply_butterworth_bandpass
from POS.pos_processing import handle_nan_values

# Constants for hr/hrv calculations
MIN_MAD_S = 0.005
K_MAD_MULTIPLIER = 3.0

def find_signal_peaks(filtered_signal, target_fps):
    if filtered_signal is None or filtered_signal.size == 0:
        return None

    if target_fps <= 0:
        return None

    min_distance_samples = max(1, int(target_fps / MAX_HR_HZ))    
    signal_std = np.std(filtered_signal)
    
    if signal_std == 0:
        return None
    
    prominence_threshold = signal_std * 0.8
    
    peaks_data = scipy.signal.find_peaks(
        filtered_signal,
        distance=min_distance_samples,
        prominence=prominence_threshold,
        # height=height_threshold
    )
    return peaks_data

def calculate_hr_hrv(peak_timestamps: np.ndarray):
    if peak_timestamps is None or peak_timestamps.size < MIN_PEAKS_FOR_HRV:
        return None

    ibis_sec = np.diff(peak_timestamps)
    
    if ibis_sec.size < MIN_PEAKS_FOR_HRV - 1:
        return None

    valid_mask_physio = (ibis_sec >= MIN_VALID_IBI_S) & (ibis_sec <= MAX_VALID_IBI_S)
    ibis_sec_physio = ibis_sec[valid_mask_physio]

    if ibis_sec_physio.size < MIN_PEAKS_FOR_HRV - 1:
        return None

    median_ibi = np.median(ibis_sec_physio)

    mad_ibi_calculated = scipy.stats.median_abs_deviation(ibis_sec_physio, scale='normal', nan_policy='omit')

    if np.isnan(mad_ibi_calculated):
        ibis_sec_final = ibis_sec_physio
    else:
        mad_ibi_for_filtering = max(mad_ibi_calculated, MIN_MAD_S)

        lower_bound = max(median_ibi - K_MAD_MULTIPLIER * mad_ibi_for_filtering, MIN_VALID_IBI_S)
        upper_bound = min(median_ibi + K_MAD_MULTIPLIER * mad_ibi_for_filtering, MAX_VALID_IBI_S)
        
        valid_mask_stat = (ibis_sec_physio >= lower_bound) & (ibis_sec_physio <= upper_bound)
        ibis_sec_final = ibis_sec_physio[valid_mask_stat]

    if ibis_sec_final.size < MIN_PEAKS_FOR_HRV - 1:
        return None

    mean_ibi_sec = np.mean(ibis_sec_final)
    
    if mean_ibi_sec == 0:
        return {'hr': np.nan, 'sdnn': np.nan, 'rmssd': np.nan, 'hrv_quality': 'error_zero_mean_ibi'}

    hr_bpm = 60.0 / mean_ibi_sec
    sdnn_ms = np.std(ibis_sec_final) * 1000.0

    if ibis_sec_final.size >= 2:
        diffs_ibi = np.diff(ibis_sec_final)
        np.square(diffs_ibi, out=diffs_ibi) 
        rmssd_ms = np.sqrt(np.mean(diffs_ibi)) * 1000.0
    else:
        rmssd_ms = np.nan
        
    if sdnn_ms > MAX_ACCEPTABLE_SDNN_MS:
        return {'hr': hr_bpm, 'sdnn': np.nan, 'rmssd': np.nan, 'hrv_quality': 'unstable_sdnn'}
    
    if not np.isnan(rmssd_ms) and rmssd_ms > MAX_ACCEPTABLE_RMSSD_MS:
        return {'hr': hr_bpm, 'sdnn': sdnn_ms, 'rmssd': np.nan, 'hrv_quality': 'unstable_rmssd'}
        
    return {'hr': hr_bpm, 'sdnn': sdnn_ms, 'rmssd': rmssd_ms, 'hrv_quality': 'ok'}

def extract_breathing_signal(rgb_buffer, timestamps):
    cleaned_rgb, cleaned_timestamps = handle_nan_values(rgb_buffer, timestamps)
    if cleaned_rgb is None or cleaned_timestamps is None:
        return None, None
    if cleaned_rgb.shape[0] < MIN_SAMPLES_FOR_BR:
        return None, None
    approx_fps = 1.0 / np.mean(np.diff(cleaned_timestamps))
    green_signal = cleaned_rgb[:, 1]
    detrended_green = scipy.signal.detrend(green_signal, type='linear')
    filtered_breathing_signal = apply_butterworth_bandpass(
        detrended_green, MIN_BR_HZ, MAX_BR_HZ, approx_fps, order=3
    )
    if filtered_breathing_signal is None:
        return None, None
    mean_green = np.mean(filtered_breathing_signal)
    std_green = np.std(filtered_breathing_signal)
    if std_green == 0:
        return None, None
    detrended_normalized_green = (filtered_breathing_signal - mean_green) / std_green
    return detrended_normalized_green, cleaned_timestamps

def calculate_breathing_rate_welch(breathing_signal, timestamps):
    if breathing_signal is None or timestamps is None or len(breathing_signal) < 2:
        return None
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        return None
    fs = 1.0 / np.median(dt)
    signal = breathing_signal - np.mean(breathing_signal)
    nperseg = len(signal)
    nfft = 4096
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg, nfft=nfft)
    valid_mask = (freqs >= MIN_BR_HZ) & (freqs <= MAX_BR_HZ)
    if not np.any(valid_mask):
        return None
    valid_freqs = freqs[valid_mask]
    valid_psd = psd[valid_mask]
    if len(valid_psd) == 0:
        return None
    peak_idx = np.argmax(valid_psd)
    dominant_freq_hz = valid_freqs[peak_idx]
    breathing_rate_bpm = dominant_freq_hz * 60.0
    return breathing_rate_bpm
