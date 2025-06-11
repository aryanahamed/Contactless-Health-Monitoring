from POS.pos_processing import select_best_pos_signal, apply_windowing
from POS.signal_extraction import (
    calculate_hr_fft, find_signal_peaks, calculate_hrv
)
from POS.smoothing import smooth_bpm_multi_stage, get_current_smoothed_bpm

def process_hr(series):
    best_filt, _, best_ts, best_fps, quality = select_best_pos_signal(series)

    last_sdnn = last_rmssd = None
    hrv_quality_status = 'N/A'
    
    if best_filt is not None:        
        fft_hr = calculate_hr_fft(best_filt, best_fps)
        peaks_tuple = find_signal_peaks(best_filt, best_fps, fft_hr)

        peaks_ts = None
        if peaks_tuple:
            peaks_indices = peaks_tuple[0]
            peaks_ts = best_ts[peaks_indices] if best_ts is not None else None
            
        hr_hrv_results = calculate_hrv(peaks_ts)
        if hr_hrv_results:
            last_sdnn = hr_hrv_results.get('sdnn')
            last_rmssd = hr_hrv_results.get('rmssd')
            hrv_quality_status = hr_hrv_results.get('hrv_quality', 'N/A')

    if fft_hr is not None:
        smoothed_hr = smooth_bpm_multi_stage(fft_hr, quality)
    else:
        smoothed_hr = get_current_smoothed_bpm()
    
    return smoothed_hr, last_sdnn, last_rmssd, quality, hrv_quality_status
