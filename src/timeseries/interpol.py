import numpy as np
import config
from scipy.interpolate import PchipInterpolator


def pchip(t_uniform, t_irregular, values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)

    if values.ndim == 1:
        values = values[:, None]

    # nan checking
    mask = ~np.any(np.isnan(values), axis=1)
    min_valid = int(0.6 * len(t_uniform))
    valid_count = np.count_nonzero(mask)

    if valid_count < min_valid:
        return np.full((len(t_uniform), values.shape[1]), np.nan, dtype=np.float32)

    # extract valid data
    t_valid = t_irregular[mask]
    v_valid = values[mask]
    w_valid = weights[mask]
    #compute weighted values
    w_expanded = w_valid[:, np.newaxis]
    weighted_values = w_expanded * v_valid
    # Create interpolates
    f_weighted = PchipInterpolator(t_valid, weighted_values, axis=0, extrapolate=False)
    f_weights = PchipInterpolator(t_valid, w_valid, extrapolate=False)

    # Find interpolation domain
    inside_mask = (t_uniform >= t_valid[0]) & (t_uniform <= t_valid[-1])
    if not np.any(inside_mask):
        return np.full((len(t_uniform), values.shape[1]), np.nan, dtype=np.float32)

    # pre allocation
    out = np.full((len(t_uniform), values.shape[1]), np.nan, dtype=np.float32)
    # Get points
    t_inside = t_uniform[inside_mask]

    # interpolate
    weighted_interp = f_weighted(t_inside)
    weights_interp = f_weights(t_inside)

    # broadcasting for division
    if weighted_interp.ndim == 1:
        weights_interp_expanded = weights_interp
    else:
        weights_interp_expanded = weights_interp[:, np.newaxis]
    nonzero_mask = weights_interp != 0
    result = np.full_like(weighted_interp, np.nan)

    if weighted_interp.ndim == 1:
        result[nonzero_mask] = weighted_interp[nonzero_mask] / weights_interp[nonzero_mask]
    else:
        np.divide(weighted_interp, weights_interp_expanded,
                  out=result, where=nonzero_mask[:, np.newaxis])

    out[inside_mask] = result

    return out.squeeze()


def get_uniform(series_dict, t_uniform_template=None):
    timestamps = series_dict["timestamps"]
    if not isinstance(timestamps, np.ndarray):
        timestamps = np.asarray(timestamps)

    # last valid timestamp
    valid_timestamps = timestamps[~np.isnan(timestamps)]
    if len(valid_timestamps) == 0:
        return None

    end_time = valid_timestamps[-1]

    # uniform time array
    if t_uniform_template is None:
        t_uniform = np.linspace(end_time - config.window, end_time,
                                int(config.window * config.hz), dtype=np.float64)
    else:
        t_uniform = t_uniform_template + end_time

    interp_series = {}

    # process each region
    for region in series_dict:
        if region in ["timestamps", "weights"]:
            continue

        values = series_dict[region]
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)

        # weights for this region
        region_weights = series_dict["weights"][region]
        if not isinstance(region_weights, np.ndarray):
            region_weights = np.asarray(region_weights)

        # interpolate
        interp_rgb = pchip(t_uniform, timestamps, values, region_weights)
        interp_series[region] = interp_rgb

    interp_series["timestamps"] = t_uniform
    return interp_series