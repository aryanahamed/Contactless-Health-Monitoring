from POS.pos_processing import select_best_pos_signal
from POS.signal_extraction import (
    find_signal_peaks, calculate_hr_hrv, extract_breathing_signal, calculate_breathing_rate_welch
)

def process_hr(series):
    best_filt, _, best_ts, best_fps, quality = select_best_pos_signal(series)

    last_hr = last_sdnn = last_rmssd = None
    hrv_quality_status = 'N/A'
    if best_filt is not None:
        peaks_tuple = find_signal_peaks(best_filt, best_fps)
        peaks_ts = None
        if peaks_tuple:
            peaks_indices = peaks_tuple[0]
            peaks_ts = best_ts[peaks_indices] if best_ts is not None else None
        hr_hrv_results = calculate_hr_hrv(peaks_ts)
        if hr_hrv_results:
            last_hr = hr_hrv_results.get('hr')
            last_sdnn = hr_hrv_results.get('sdnn')
            last_rmssd = hr_hrv_results.get('rmssd')
            hrv_quality_status = hr_hrv_results.get('hrv_quality', 'N/A')
            # print(f"HR/HRV Results: HR={last_hr}, SDNN={last_sdnn}, RMSSD={last_rmssd}, Quality={hrv_quality_status}")
        else:
            print("Could not calculate HR/HRV from peaks.")
    return last_hr, last_sdnn, last_rmssd, quality, hrv_quality_status


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
