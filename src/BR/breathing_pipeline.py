import numpy as np
from POS.signal_processing import select_best_signal
from BR.br_extraction import extract_breathing_signal, calculate_breathing_rate_welch
from BR.smoothing_br import smooth_br_multi_stage

def process_breathing(best_signal, best_ts, fps, quality):
    # This is the main pipeline for processing breathing signal and extracting breathing rate
    last_br = None

    if best_signal is not None and best_ts is not None and len(best_ts) > 1:
        fs = fps  # sampling frequency
        window_center_time = np.mean(best_ts)  # center time of the window

        # process the signal
        br_signal, br_ts = extract_breathing_signal(best_signal, best_ts, fs)

        if br_signal is not None and br_ts is not None:
            # Estimate breathing rate
            raw_br = calculate_breathing_rate_welch(br_signal, br_ts)
            # Smoothen
            smoothed_br = smooth_br_multi_stage(raw_br, timestamp=window_center_time, quality_score=quality)

            if smoothed_br is not None:
                last_br = smoothed_br
        
    return last_br