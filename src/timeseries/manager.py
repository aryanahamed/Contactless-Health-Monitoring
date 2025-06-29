import config
import numpy as np
from timeseries.interpol import get_uniform


class TimeSeries:
    def __init__(self):
        self.regions = config.regions
        self.max_samples = config.buffer_size

        # Circular buffers
        self.times = np.full(self.max_samples, np.nan, dtype=np.float64)
        self.data = {r: np.full((self.max_samples, 3), np.nan, dtype=np.float32) for r in self.regions}
        self.weights = {r: np.full(self.max_samples, np.nan, dtype=np.float32) for r in self.regions}
        self._temp_times = np.empty(self.max_samples, dtype=np.float64)
        self._temp_data = {r: np.empty((self.max_samples, 3), dtype=np.float32) for r in self.regions}
        self._temp_weights = {r: np.empty(self.max_samples, dtype=np.float32) for r in self.regions}

        self.next_slot = 0
        self.num_samples = 0
        self.time_template = np.linspace(-config.window, 0, int(config.window * config.hz), dtype=np.float64)
        #cache
        self._nan_rgb = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    def get(self, roi_patch, timestamp):
        #store the frame data
        self.times[self.next_slot] = timestamp

        for region in self.regions:
            region_patch_data = roi_patch.get(region)
            if region_patch_data is not None:
                self.data[region][self.next_slot] = region_patch_data["rgb"]
                self.weights[region][self.next_slot] = region_patch_data["weight"]
            else:
                #nan arrays
                self.data[region][self.next_slot] = self._nan_rgb
                self.weights[region][self.next_slot] = np.nan

        # update the buffer
        self.next_slot = (self.next_slot + 1) % self.max_samples
        self.num_samples = min(self.num_samples + 1, self.max_samples)

        if self.num_samples < 2:
            return None

        # ordering chronologically
        current_times, current_data, current_weights = self._get_ordered_data()

        # validating early
        if self.num_samples < self.max_samples:
            valid_mask = ~np.isnan(current_times[:self.num_samples])
            if np.sum(valid_mask) < 2:
                return None
            valid_times = current_times[:self.num_samples][valid_mask]
        else:
            valid_mask = ~np.isnan(current_times)
            if np.sum(valid_mask) < 2:
                return None
            valid_times = current_times[valid_mask]

        if (valid_times[-1] - valid_times[0]) < config.window:
            return None

        #build the dict
        series_dict = {
            "timestamps": current_times,
            "weights": current_weights
        }
        series_dict.update(current_data)

        return get_uniform(series_dict, self.time_template)

    def _get_ordered_data(self):
        if self.num_samples < self.max_samples:
            return (self.times[:self.num_samples],
                    {r: self.data[r][:self.num_samples] for r in self.regions},
                    {r: self.weights[r][:self.num_samples] for r in self.regions})
        else:
            if self.next_slot == 0:
                return self.times, self.data, self.weights
            else:
                end_size = self.max_samples - self.next_slot
                self._temp_times[:end_size] = self.times[self.next_slot:]
                self._temp_times[end_size:] = self.times[:self.next_slot]
                # copy data
                for region in self.regions:
                    self._temp_data[region][:end_size] = self.data[region][self.next_slot:]
                    self._temp_data[region][end_size:] = self.data[region][:self.next_slot]
                    self._temp_weights[region][:end_size] = self.weights[region][self.next_slot:]
                    self._temp_weights[region][end_size:] = self.weights[region][:self.next_slot]

                return self._temp_times, self._temp_data, self._temp_weights



