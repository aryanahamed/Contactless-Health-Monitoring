import numpy as np
import scipy.signal
from numba import njit
from config import MIN_SAMPLES_FOR_POS

@njit(cache=True)
def _compute_chrom_core(normalized_rgb):
    n_samples = normalized_rgb.shape[0]
    X_chrom = np.empty(n_samples, dtype=np.float64)
    Y_chrom = np.empty(n_samples, dtype=np.float64)
    
    for i in range(n_samples):
        r, g, b = normalized_rgb[i, 0], normalized_rgb[i, 1], normalized_rgb[i, 2]
        X_chrom[i] = 3 * r - 2 * g
        Y_chrom[i] = 1.5 * r + g - 1.5 * b
    
    std_X = np.std(X_chrom)
    std_Y = np.std(Y_chrom)
    
    if std_X == 0:
        return Y_chrom
    if std_Y == 0:
        return X_chrom
    
    alpha = std_X / std_Y
    S_chrom = X_chrom - alpha * Y_chrom
    return S_chrom

def apply_chrom_projection(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0] < MIN_SAMPLES_FOR_POS:
        return None

    mean_rgb = np.mean(rgb_buffer, axis=0)
    if np.any(mean_rgb <= 1e-10):
        return None
    
    if np.any(np.std(rgb_buffer, axis=0) < 1e-6):
        return None
    
    normalized_rgb = rgb_buffer / mean_rgb
    raw_chrom_signal = _compute_chrom_core(normalized_rgb)
    
    detrended_chrom_signal = scipy.signal.detrend(raw_chrom_signal, axis=0, overwrite_data=True)

    if detrended_chrom_signal is not None and np.std(detrended_chrom_signal) < 1e-8:
        return None

    return detrended_chrom_signal