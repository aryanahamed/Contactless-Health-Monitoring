import numpy as np
import scipy.signal
import scipy.stats
from numba import njit
from config import (
    MAX_HR_HZ,
    MIN_PEAKS_FOR_HRV,
    MIN_VALID_IBI_S,
    MAX_VALID_IBI_S,
    MAX_ACCEPTABLE_SDNN_MS,
    MAX_ACCEPTABLE_RMSSD_MS,
    BAND_MIN_HZ,
    BAND_MAX_HZ,
)

MIN_MAD_S = 0.005
K_MAD_MULTIPLIER = 2.5


@njit(cache=True)
def _quadratic_interpolation(y_values, peak_index):
    if peak_index == 0 or peak_index == len(y_values) - 1:
        return float(peak_index)

    y_minus_1 = y_values[peak_index - 1]
    y_0 = y_values[peak_index]
    y_plus_1 = y_values[peak_index + 1]

    p = (y_minus_1 - y_plus_1) / (2 * (y_minus_1 - 2 * y_0 + y_plus_1))
    return float(peak_index) + p


def calculate_hr_fft(filtered_signal, fps):
    if filtered_signal is None or len(filtered_signal) < 128:
        return None
        
    if fps <= 0:
        return None
    
    try:
        nperseg = len(filtered_signal)
        
        freqs, psd = scipy.signal.welch(
            filtered_signal, 
            fs=fps, 
            nperseg=nperseg,
            window='hann'
        )
                
        valid_mask = (freqs >= BAND_MIN_HZ) & (freqs <= BAND_MAX_HZ)
        
        if not np.any(valid_mask):
            return None
            
        valid_freqs = freqs[valid_mask]
        valid_psd = psd[valid_mask]
        
        if len(valid_psd) == 0:
            return None

        peak_idx_in_valid = np.argmax(valid_psd)
        
        interp_peak_idx = _quadratic_interpolation(valid_psd, peak_idx_in_valid)
        
        if interp_peak_idx > 0 and interp_peak_idx < len(valid_freqs) - 1:
            f1, f2 = valid_freqs[int(interp_peak_idx)], valid_freqs[int(interp_peak_idx) + 1]
            dominant_freq_hz = f1 + (f2 - f1) * (interp_peak_idx - int(interp_peak_idx))
        else:
            dominant_freq_hz = valid_freqs[peak_idx_in_valid]

        estimated_hr_bpm = dominant_freq_hz * 60.0
        
        if 40 <= estimated_hr_bpm <= 180:
            return estimated_hr_bpm
        else:
            return None
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def find_signal_peaks(filtered_signal, target_fps, expected_hr_bpm=None):
    if filtered_signal is None or filtered_signal.size == 0:
        return None

    if target_fps <= 0:
        return None    
    if expected_hr_bpm is not None and 40 <= expected_hr_bpm <= 180:
        expected_period_sec = 60.0 / expected_hr_bpm
        min_distance_samples = max(1, int(target_fps * expected_period_sec * 0.7))
    else:
        min_distance_samples = max(1, int(target_fps / MAX_HR_HZ))    
    signal_std = np.std(filtered_signal)
    
    if signal_std == 0:
        return None
    
    prominence_threshold = signal_std * 0.2
    height_threshold = signal_std * 0.1
    
    peaks_data = scipy.signal.find_peaks(
        filtered_signal,
        distance=min_distance_samples,
        prominence=prominence_threshold,
        height=height_threshold
    )
    
    return peaks_data

@njit(cache=True)
def _calculate_hrv_core(ibis_sec_final):
    mean_ibi_sec = np.mean(ibis_sec_final)
    sdnn_ms = np.std(ibis_sec_final) * 1000.0
    if ibis_sec_final.size >= 2:
        diffs_ibi = np.diff(ibis_sec_final)
        diffs_ibi = np.square(diffs_ibi)
        rmssd_ms = np.sqrt(np.mean(diffs_ibi)) * 1000.0
    else:
        rmssd_ms = np.nan
    return mean_ibi_sec, sdnn_ms, rmssd_ms

def calculate_hrv(peak_timestamps):
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

    mean_ibi_sec, sdnn_ms, rmssd_ms = _calculate_hrv_core(ibis_sec_final)

    if mean_ibi_sec == 0:
        return {'sdnn': np.nan, 'rmssd': np.nan, 'hrv_quality': 'error_zero_mean_ibi'}

    if sdnn_ms > MAX_ACCEPTABLE_SDNN_MS:
        return {'sdnn': np.nan, 'rmssd': np.nan, 'hrv_quality': 'unstable_sdnn'}

    if not np.isnan(rmssd_ms) and rmssd_ms > MAX_ACCEPTABLE_RMSSD_MS:
        return {'sdnn': sdnn_ms, 'rmssd': np.nan, 'hrv_quality': 'unstable_rmssd'}

    return {'sdnn': sdnn_ms, 'rmssd': rmssd_ms, 'hrv_quality': 'ok'}

