from collections import deque
import core.config as config
import numpy as np
from .spatial_average import sa_region

class Manager:
    def __init__(self):
        self.region_name = config.regions
        self.buffer_size = config.buffer_size
        self.series = {
            region: deque(maxlen=self.buffer_size)
            for region in self.region_name
        }

    def get_series(self, patches,timestamp):
        for region in self.region_name:
            if region in patches:
                rgb = sa_region(patches[region])
                self.series[region].append((rgb, timestamp))
            else:
                self.series[region].append((np.array([np.nan, np.nan, np.nan]), timestamp))

        if all(len(self.series[r]) == self.buffer_size for r in self.region_name):
            return self.get_full()
        else:
            return None

    def get_full(self):
        output = {}
        first_region = next(iter(self.series))
        _, times = zip(*self.series[first_region])
        output["timestamps"] = np.array(times)
        for region,queue in self.series.items():
            rgbs = [rgb for rgb, _ in queue]
            output[region] = np.array(rgbs)

        return output