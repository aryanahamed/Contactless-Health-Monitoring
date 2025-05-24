from collections import deque
import config
import numpy as np
from timeseries.interpol import get_uniform_series

class Manager:
    def __init__(self):
        self.region_name = config.regions
        self.buffer_size = config.buffer_size + 60  # Large enough for more than 8 sec of data
        self.series = {
            **{region: deque(maxlen=self.buffer_size) for region in self.region_name},
            "timestamps": deque(maxlen=self.buffer_size)
        }
        self.t_uniform_template = np.linspace(
            -config.window, 0, int(config.window * config.hz)
        )

    def get_series(self, patches, timestamp):
        self.series["timestamps"].append(timestamp)

        for region in self.region_name:
            rgb = np.array([np.nan, np.nan, np.nan])  # Default to NaN

            if region in patches:
                rgb_candidate = sa_region(patches[region])
                if not np.isnan(rgb_candidate).any():
                    rgb = rgb_candidate

            self.series[region].append(rgb)

        # Check if we have enough time span before interpolating
        timestamps = self.series["timestamps"]
        if len(timestamps) < 2:
            return None

        buffer_duration = timestamps[-1] - timestamps[0]
        if buffer_duration < config.window:
            return None  # Wait until we span at least w seconds

        # Prepare data for interpolation
        series_np = {k: np.array(list(v)) for k, v in self.series.items()}
        return get_uniform_series(series_np, self.t_uniform_template)

def sa_region(patch):
    if patch is None:
        return np.array([np.nan, np.nan, np.nan])

    flat = patch.reshape(-1, 3)
    if len(flat) == 0:
        return np.array([np.nan, np.nan, np.nan])

    mean_bgr = np.mean(flat, axis=0)
    return mean_bgr[::-1]  # Convert BGR to RGB



