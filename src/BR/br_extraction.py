import numpy as np
import scipy.signal
from scipy.signal import find_peaks

from config import (
    MIN_BR_HZ,
    MAX_BR_HZ,
    MIN_SAMPLES_FOR_BR,
    MIN_HR_HZ,
    MAX_HR_HZ,
    RESP_SIG_INTERPOLATION_FS
)
from POS.pos_processing import apply_butterworth_bandpass


def extract_breathing_signal(rppg_signal, timestamps, fs):
    if rppg_signal is None or len(rppg_signal) < MIN_SAMPLES_FOR_BR:
        return None, None

    cardiac_filtered_ppg = apply_butterworth_bandpass(
        rppg_signal, MIN_HR_HZ, MAX_HR_HZ, fs, order=2
    )
    if cardiac_filtered_ppg is None:
        return None, None


    peak_distance = int(fs / MAX_HR_HZ)
    peaks, _ = find_peaks(cardiac_filtered_ppg, height=np.std(cardiac_filtered_ppg) * 0.4, distance=peak_distance)

    if len(peaks) < 5:
        return None, None

    peak_times = timestamps[peaks]


    am_signal = rppg_signal[peaks]

    ibi_signal = np.diff(peak_times)
    ibi_times = peak_times[1:]

    if len(am_signal) < 2 or len(ibi_signal) < 2:
        return None, None

    interp_time = np.arange(timestamps[0], timestamps[-1], 1.0 / RESP_SIG_INTERPOLATION_FS)
    
    interp_am = np.interp(interp_time, peak_times, am_signal)
    interp_ibi = np.interp(interp_time, ibi_times, ibi_signal)
    
    am_z = (interp_am - np.mean(interp_am)) / np.std(interp_am)
    ibi_z = (interp_ibi - np.mean(interp_ibi)) / np.std(interp_ibi)
    
    fused_signal = scipy.signal.detrend(am_z + ibi_z)
    
    return fused_signal, interp_time


def calculate_breathing_rate_welch(breathing_signal, timestamps):
    if breathing_signal is None or timestamps is None or len(breathing_signal) < 2:
        return None
    
    fs = 1.0 / np.median(np.diff(timestamps))
    if fs <= 0 or np.isnan(fs):
        return None

    signal = breathing_signal - np.mean(breathing_signal)
    
    nperseg = min(len(signal), 256)
    nfft = 4096

    try:
        freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg, nfft=nfft, window='hann')
    except ValueError:
        return None

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