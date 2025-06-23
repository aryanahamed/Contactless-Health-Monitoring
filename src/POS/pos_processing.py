import numpy as np
import scipy.signal
from numba import njit
from config import MIN_SAMPLES_FOR_POS

@njit(cache=True)
def _compute_pos_core(normalized_rgb):
    n_samples = normalized_rgb.shape[0]
    X = np.empty(n_samples, dtype=np.float32)
    Y = np.empty(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = normalized_rgb[i, 1] - normalized_rgb[i, 2]  # G - B
        Y[i] = normalized_rgb[i, 1] + normalized_rgb[i, 2] - 2 * normalized_rgb[i, 0]  # G + B - 2R
    
    std_X = np.std(X)
    std_Y = np.std(Y)
    
    if std_Y == 0:
        return None
    
    alpha = std_X / std_Y
    S = X + alpha * Y
    return S

def apply_pos_projection(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0] < MIN_SAMPLES_FOR_POS:
        return None

    mean_rgb = np.mean(rgb_buffer, axis=0)
    if np.any(mean_rgb <= 1e-10):
        return None
    
    if np.any(np.std(rgb_buffer, axis=0) < 1e-6):
        return None
    
    normalized_rgb = rgb_buffer / mean_rgb
    raw_pos_signal = _compute_pos_core(normalized_rgb)
    
    detrended_pos_signal = scipy.signal.detrend(raw_pos_signal, axis=0, overwrite_data=True)

    if detrended_pos_signal is not None and np.std(detrended_pos_signal) < 1e-8:
        return None

    return detrended_pos_signal