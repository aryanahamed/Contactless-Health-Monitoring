from POS.pos_processing import select_best_pos_signal, apply_windowing
from POS.signal_extraction import (
    estimate_hr_fft, find_signal_peaks, calculate_hr_hrv, extract_breathing_signal, calculate_breathing_rate_welch
)
from POS.smoothing import smooth_bpm_multi_stage, get_current_smoothed_bpm

def process_hr(series):
    best_filt, _, best_ts, best_fps, quality = select_best_pos_signal(series)

    raw_hr = last_sdnn = last_rmssd = None
    hrv_quality_status = 'N/A'
    
    if best_filt is not None:        
        fft_hr_estimate = estimate_hr_fft(best_filt, best_fps)
        peaks_tuple = find_signal_peaks(best_filt, best_fps, fft_hr_estimate)

        peaks_ts = None
        if peaks_tuple:
            peaks_indices = peaks_tuple[0]
            peaks_ts = best_ts[peaks_indices] if best_ts is not None else None
            
        # hr_hrv_results = calculate_hr_hrv(peaks_ts)
        # if hr_hrv_results:
        #     raw_hr = hr_hrv_results.get('hr')
        #     last_sdnn = hr_hrv_results.get('sdnn')
        #     last_rmssd = hr_hrv_results.get('rmssd')
        #     hrv_quality_status = hr_hrv_results.get('hrv_quality', 'N/A')
            # print(f"Peak-based HR: {raw_hr:.1f} BPM, HRV Quality: {hrv_quality_status}")
        # else:
        #     print("Could not calculate HR/HRV from peaks.")
        #     if fft_hr_estimate is not None:
        #         raw_hr = fft_hr_estimate
        #         print(f"Using FFT estimate as fallback: {raw_hr:.1f} BPM")
        raw_hr = fft_hr_estimate
    
    if raw_hr is not None:
        smoothed_hr = smooth_bpm_multi_stage(raw_hr, quality)
    else:
        smoothed_hr = get_current_smoothed_bpm()
    
    return smoothed_hr, last_sdnn, last_rmssd, quality, hrv_quality_status


def process_breathing(series):
    _, best_rgb, best_ts, _, _ = select_best_pos_signal(series)
    last_br = None
    if best_rgb is not None and best_ts is not None:
        br_signal, br_ts = extract_breathing_signal(best_rgb, best_ts)
        current_br = calculate_breathing_rate_welch(br_signal, br_ts)
        if current_br is not None:
            last_br = current_br
            print(f"BR Result (updated): {last_br:.1f} BPM")
        else:
            print("Could not calculate BR in this interval.")
    return last_br
