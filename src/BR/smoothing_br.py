import numpy as np
from collections import deque
from numba import njit
# BR Globals
# Buffers and state for BR filtering
_br_history = deque(maxlen=30)
_br_quality_history = deque(maxlen=30)
_br_median_buffer = deque(maxlen=5)
_br_ema_value = None
_br_last_timestamp = None
_br_last_valid_br = 15.0

# BR Configs
BR_OUTLIER_WINDOW_SIZE = 8
BR_MEDIAN_WINDOW_SIZE = 5
BR_EMA_ALPHA = 0.2
MAX_BR_CHANGE_PER_SEC = 2
BR_MIN_QUALITY_THRESHOLD = 1.5
PHYSIO_MIN_BR = 8
PHYSIO_MAX_BR = 35
BR_MIN_MAD_FOR_Z_SCORE_CALC = 0.5
BR_OUTLIER_WINDOW_SIZE = 8

def reset_br_filters():
    # Reset all BR filter states to initial values
    global _br_history, _br_quality_history, _br_median_buffer, _br_ema_value, _br_last_timestamp, _br_last_valid_br
    _br_history.clear()
    _br_quality_history.clear()
    _br_median_buffer.clear()
    _br_ema_value = None
    _br_last_timestamp = None
    _br_last_valid_br = 15.0


def reject_br_outliers(new_br, current_timestamp, quality_score):
    # Rejects low quality BR values
    global _br_last_timestamp, _br_last_valid_br, _br_history

    if new_br is None or np.isnan(new_br):
        return None

    is_establishing_baseline = len(_br_history) == 0
    required_quality = 3.0 if is_establishing_baseline else BR_MIN_QUALITY_THRESHOLD

    if quality_score < required_quality:
        return None

    if new_br < PHYSIO_MIN_BR or new_br > PHYSIO_MAX_BR:
        return None

    # Limit sudden changes in BR based on time difference
    if _br_last_timestamp is not None and _br_last_valid_br is not None:
        time_diff = current_timestamp - _br_last_timestamp
        if time_diff > 0:
            max_allowed_change = MAX_BR_CHANGE_PER_SEC * time_diff
            if abs(new_br - _br_last_valid_br) > max_allowed_change:
                return None

    # Outlier rejection using modified z-score
    if len(_br_history) >= 3:
        recent_values = np.array(list(_br_history)[-BR_OUTLIER_WINDOW_SIZE:])
        if len(recent_values) >= 3:
            median_recent = np.median(recent_values)
            mad = np.median(np.abs(recent_values - median_recent))
            if mad >= BR_MIN_MAD_FOR_Z_SCORE_CALC:
                modified_z_score = 0.6745 * (new_br - median_recent) / mad
                # Can tweek this value if needed
                if abs(modified_z_score) > 2.5:
                    return None
        
    _br_last_timestamp = current_timestamp
    _br_last_valid_br = new_br
    return new_br

@njit(cache=True)
def _median_numba_br(arr):
    return np.median(arr)

def add_to_br_median_filter(br_value):
    # Maintain a rolling median filter for BR values
    global _br_median_buffer
    if br_value is None:
        if len(_br_median_buffer) > 0:
            return _median_numba_br(np.array(_br_median_buffer))
        return None
    _br_median_buffer.append(br_value)
    return _median_numba_br(np.array(_br_median_buffer))

@njit(cache=True)
def _ema_numba_br(prev_ema, value, alpha):
    return alpha * value + (1 - alpha) * prev_ema

def apply_br_exponential_smoothing(median_br):
    # Applies ema smoothing to the median BR
    global _br_ema_value
    if median_br is None:
        return _br_ema_value
    if _br_ema_value is None:
        _br_ema_value = median_br
        return _br_ema_value
    _br_ema_value = _ema_numba_br(_br_ema_value, median_br, BR_EMA_ALPHA)
    return _br_ema_value

def smooth_br_multi_stage(new_br, timestamp, quality_score=1.0):
    # This is the main entry
    global _br_history, _br_quality_history, _br_ema_value

    filtered_br = reject_br_outliers(new_br, timestamp, quality_score)
    
    if filtered_br is not None:
        _br_history.append(filtered_br)
        _br_quality_history.append(quality_score)
        
        median_br = add_to_br_median_filter(filtered_br)
        final_br = apply_br_exponential_smoothing(median_br)
    else:
        final_br = _br_ema_value

    return final_br

def get_current_smoothed_br():
    # Returns the latest smoothed BR value
    return _br_ema_value

def get_br_filter_status():
    # Returns diagnostic infos
    return {
        'br_history_length': len(_br_history),
        'br_median_buffer_length': len(_br_median_buffer),
        'current_br_ema': _br_ema_value,
        'last_valid_br': _br_last_valid_br
    }
