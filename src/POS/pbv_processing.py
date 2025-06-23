import numpy as np
from numba import njit
import scipy.signal

@njit(cache=True)
def _pbv_core(normalized_rgb):
    cov_matrix = np.cov(normalized_rgb.T)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    principal_direction = eigvecs[:, np.argmax(eigvals)]
    pbv_signal = np.dot(normalized_rgb, principal_direction)
    return pbv_signal

def apply_pbv_projection(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0] < 60:
        return None

    mean_rgb = np.mean(rgb_buffer, axis=0)
    if np.any(mean_rgb <= 1e-10):
        return None

    normalized_rgb = rgb_buffer / mean_rgb
    normalized_rgb -= np.mean(normalized_rgb, axis=0)

    pbv_signal = _pbv_core(normalized_rgb)
    pbv_signal = scipy.signal.detrend(pbv_signal, axis=0, overwrite_data=True)

    if np.std(pbv_signal) < 1e-8:
        return None

    return pbv_signal
