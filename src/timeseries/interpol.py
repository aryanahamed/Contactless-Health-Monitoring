import numpy as np
import config
from scipy.interpolate import PchipInterpolator

def pchip(t_uniform, t_irregular, values):
    """
    Performs PCHIP interpolation with extrapolation disabled.
    Returns NaNs outside the valid interpol range or if not enough valid points.
    """
    min_valid = int(0.5 * len(t_uniform))  #  at least 50p valid frames
    mask = ~np.isnan(values)
    if np.count_nonzero(mask) < min_valid:
        return np.full(t_uniform.shape, np.nan, dtype=np.float32)

    t_valid = t_irregular[mask]
    v_valid = values[mask]

    f = PchipInterpolator(t_valid, v_valid, extrapolate=False)
    out = np.full(t_uniform.shape, np.nan, dtype=np.float32)

    inside = (t_uniform >= t_valid[0]) & (t_uniform <= t_valid[-1])
    out[inside] = f(t_uniform[inside])
    return out



def get_uniform_series(series_dict, t_uniform_template=None):
    timestamps = np.array(series_dict["timestamps"])
    end_time = timestamps[-1]

    if t_uniform_template is None:
        t_uniform = np.linspace(end_time - config.window, end_time, int(config.window * config.hz))
    else:
        t_uniform = t_uniform_template + end_time

    interp_series = {}
    for region, values in series_dict.items():
        if region == "timestamps":
            continue
        values = np.array(values)
        interp_rgb = np.full((len(t_uniform), 3), np.nan)
        for ch in range(3):
            interp_rgb[:, ch] = pchip(t_uniform, timestamps, values[:, ch])
        interp_series[region] = interp_rgb

    interp_series["timestamps"] = t_uniform
    return interp_series