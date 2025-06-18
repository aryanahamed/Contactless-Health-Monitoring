import numpy as np
from POS.pos_processing import select_best_pos_signal
from BR.br_extraction import extract_breathing_signal, calculate_breathing_rate_welch
from BR.smoothing_br import smooth_br_multi_stage

def process_breathing(series):
    best_pos_signal, best_rgb, best_ts, fps, quality = select_best_pos_signal(series)
    last_br = None
    
    if best_pos_signal is not None and best_ts is not None and len(best_ts) > 1:
        fs = 1.0 / np.median(np.diff(best_ts)) if fps is None else fps
        window_center_time = np.mean(best_ts)
        
        rppg_signal_1d = best_pos_signal
        br_signal, br_ts = extract_breathing_signal(rppg_signal_1d, best_ts, fs)
        
        if br_signal is not None and br_ts is not None:
            raw_br = calculate_breathing_rate_welch(br_signal, br_ts)
            smoothed_br = smooth_br_multi_stage(raw_br, timestamp=window_center_time, quality_score=quality)

            if smoothed_br is not None:
                last_br = smoothed_br
    
    return last_br