import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot

class ProcessingWorker(QThread):
    new_frame = pyqtSignal(np.ndarray)
    new_metrics = pyqtSignal(dict)

    def __init__(self, logic_function, parent=None):
        super().__init__(parent)
        self.logic_function = logic_function
        self._is_running = True

    def run(self):
        print("Worker thread started")
        self.logic_function(
            emit_frame=self.new_frame.emit,
            emit_metrics=self.new_metrics.emit,
            should_stop=lambda: not self._is_running
        )
        print("Worker thread finished")

    def stop(self):
        print("Requesting worker to stop")
        self._is_running = False


class PlotUpdateWorker(QThread):
    plot_data_ready = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_running = True
        self.start_time = None
        
        self.hr_history = []
        self.br_history = []
        self.sdnn_history = []
        self.rmssd_history = []

    @pyqtSlot(dict)
    def process_data_point(self, data_point):
        if not self._is_running:
            return
            
        try:
            timestamp = data_point.get("timestamp")
            if timestamp is None:
                return

            if self.start_time is None:
                self.start_time = timestamp

            hr_val = data_point.get("hr", {}).get("value")
            if hr_val is not None:
                self.hr_history.append((timestamp, hr_val))

            br_val = data_point.get("br", {}).get("value")
            if br_val is not None:
                self.br_history.append((timestamp, br_val))
            
            sdnn_val = data_point.get("sdnn", {}).get("value")
            if sdnn_val is not None and sdnn_val != 0:
                self.sdnn_history.append((timestamp, sdnn_val))
            
            rmssd_val = data_point.get("rmssd", {}).get("value")
            if rmssd_val is not None and rmssd_val != 0:
                self.rmssd_history.append((timestamp, rmssd_val))
            
            plot_data = self._process_plot_arrays(data_point)
            
            self.plot_data_ready.emit(plot_data)
            
        except Exception as e:
            print(f"PlotUpdateWorker error: {e}")

    def _process_plot_arrays(self, data_point):
        def get_plot_arrays(data_list):
            if not data_list:
                return np.array([]), np.array([])
            
            times = np.array([item[0] for item in data_list]) - self.start_time
            values = np.array([item[1] for item in data_list])
            return times, values
        
        rppg_signal_data = data_point.get("rppg_signal")
        rppg_times, rppg_values = (np.array([]), np.array([]))
        if rppg_signal_data and rppg_signal_data.get("timestamps") is not None:
            if rppg_signal_data["timestamps"] is not None and rppg_signal_data["values"] is not None:
                rppg_times = np.array(rppg_signal_data["timestamps"]) - self.start_time
                rppg_values = np.array(rppg_signal_data["values"])

        hr_times, hr_values = get_plot_arrays(self.hr_history)
        br_times, br_values = get_plot_arrays(self.br_history)
        sdnn_times, sdnn_values = get_plot_arrays(self.sdnn_history)
        rmssd_times, rmssd_values = get_plot_arrays(self.rmssd_history)
        
        current_elapsed_time = hr_times[-1] if hr_times.size > 0 else 0
        
        return {
            'rppg_signal_data': (rppg_times, rppg_values),
            'hr_data': (hr_times, hr_values),
            'br_data': (br_times, br_values),
            'sdnn_data': (sdnn_times, sdnn_values),
            'rmssd_data': (rmssd_times, rmssd_values),
            'current_time': current_elapsed_time
        }

    @pyqtSlot()
    def reset_data(self):
        self.start_time = None
        self.hr_history.clear()
        self.br_history.clear()
        self.sdnn_history.clear()
        self.rmssd_history.clear()

    def stop(self):
        print("Requesting plot worker to stop")
        self._is_running = False
        self.quit()
        self.wait()