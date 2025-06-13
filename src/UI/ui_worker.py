import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

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