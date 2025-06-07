import numpy as np
import time
from collections import deque

# Globals
_bpm_history = deque(maxlen=30)
_quality_history = deque(maxlen=30)
_median_buffer = deque(maxlen=7)
_ema_value = None
_last_timestamp = None
_last_valid_bpm = 70.0

# Configs
OUTLIER_WINDOW_SIZE = 8
MEDIAN_WINDOW_SIZE = 7
EMA_ALPHA = 0.05  # More smoothing = lower alpha
MAX_BPM_CHANGE_PER_SEC = 6
MIN_QUALITY_THRESHOLD = 1.2
PHYSIO_MIN_BPM = 40
PHYSIO_MAX_BPM = 180
MIN_MAD_FOR_Z_SCORE_CALC = 0.5

def reset_all_filters():
    global _bpm_history, _quality_history, _median_buffer, _ema_value, _last_timestamp, _last_valid_bpm
    _bpm_history.clear()
    _quality_history.clear()
    _median_buffer.clear()
    _ema_value = None
    _last_timestamp = None
    _last_valid_bpm = 70.0

def reject_outliers(new_bpm, quality_score):
    global _last_timestamp, _last_valid_bpm
    
    if new_bpm is None or np.isnan(new_bpm):
        return None
    
    if quality_score < MIN_QUALITY_THRESHOLD:
        return None
    
    if new_bpm < PHYSIO_MIN_BPM or new_bpm > PHYSIO_MAX_BPM:
        print(f"Rejecting Crazy BPM: {new_bpm:.1f}")
        return None
    
    current_time = time.time()
    
    startup_phase = len(_bpm_history) < 5
    
    if _last_timestamp is not None and _last_valid_bpm is not None and not startup_phase:
        time_diff = current_time - _last_timestamp
        if time_diff > 0:
            max_allowed_change = MAX_BPM_CHANGE_PER_SEC * time_diff
            actual_change = abs(new_bpm - _last_valid_bpm)
            if actual_change > max_allowed_change:
                print(f"Rejecting Crazy change: {actual_change:.1f} > {max_allowed_change:.1f}")
                return None
    
    if len(_bpm_history) >= 3 and not startup_phase:
        recent_values = list(_bpm_history)[-5:]
        median_recent = np.median(recent_values)
        mad = np.median(np.abs(np.array(recent_values) - median_recent))
        if mad >= MIN_MAD_FOR_Z_SCORE_CALC:  
            modified_z_score = 0.6745 * (new_bpm - median_recent) / mad
            if abs(modified_z_score) > 1.5:
                return None
    
    _last_timestamp = current_time
    _last_valid_bpm = new_bpm
    
    return new_bpm

from numba import njit

@njit(cache=True)
def _median_numba(arr):
    return np.median(arr)

def add_to_median_filter(bpm_value):
    global _median_buffer

    if bpm_value is None:
        if len(_median_buffer) > 0:
            return _median_numba(np.array(_median_buffer))
        return None

    _median_buffer.append(bpm_value)
    return _median_numba(np.array(_median_buffer))

@njit(cache=True)
def _ema_numba(prev_ema, value, alpha):
    return alpha * value + (1 - alpha) * prev_ema

def apply_exponential_smoothing(median_bpm):
    global _ema_value

    if median_bpm is None:
        return _ema_value

    if _ema_value is None:
        _ema_value = median_bpm
        return _ema_value

    _ema_value = _ema_numba(_ema_value, median_bpm, EMA_ALPHA)
    return _ema_value

def smooth_bpm_multi_stage(new_bpm, quality_score=1.0):
    global _bpm_history, _quality_history    
    filtered_bpm = reject_outliers(new_bpm, quality_score)
    
    if filtered_bpm is not None:
        _bpm_history.append(filtered_bpm)
        _quality_history.append(quality_score)
    
    median_bpm = add_to_median_filter(filtered_bpm)
    
    final_bpm = apply_exponential_smoothing(median_bpm)
    
    return final_bpm

def get_current_smoothed_bpm():
    return _ema_value

def get_filter_status():
    return {
        'history_length': len(_bpm_history),
        'median_buffer_length': len(_median_buffer),
        'current_ema': _ema_value,
        'last_valid_bpm': _last_valid_bpm
    }

def smooth_bpm_moving_average(new_bpm, quality_score=1.0, window_size=5):
    return smooth_bpm_multi_stage(new_bpm, quality_score)
