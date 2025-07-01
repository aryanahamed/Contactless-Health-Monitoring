import threading
import queue
import cv2
import time
import os


class CaptureThread:
    # not sure if to not limit queue size, might cause series memory issues
    ###using big mem for dataset loading only
    def __init__(self, camera_id=0, debug=False):
        self.camera_source = camera_id
        self.is_video_file = isinstance(self.camera_source, str)
        # initialize the thread
        if os.name == 'nt' and not self.is_video_file:  # Windows - use DirectShow for reliability ONLY WHEN LIVE CAMERA
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(camera_id)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise IOError
        self.use_video = isinstance(camera_id, str)  # vid
        self.queue = queue.Queue(maxsize=2000) ##set this to one in windows,driver is shit
        self.running = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)  # daemon true just allows main to exit
        self.debug = debug

    def start(self):
        self.running.set()
        self.thread.start()

    def stop(self):
        self.running.clear()
        self.thread.join()
        self.cap.release()

    def _capture_loop(self):

        while self.running.is_set():
            start = time.perf_counter()

            success, frame = self.cap.read()

            if not success:
                if self.use_video and self._check_video_end():
                    break
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            timestamp = self._get_timestamp()

            try:
                self.queue.put((frame, timestamp), timeout=0.01)
            except queue.Full:
                pass

            self._debug_print()

    ###for dataset testing only using this
    def _check_video_end(self):
        current = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if current >= total - 1:
            self.running.clear()
            return True
        return False

    def _get_timestamp(self):
        if self.use_video:
            return self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return time.perf_counter()

    ##for debugiing the mem n queue size
    def _debug_print(self):
        import psutil
        if self.debug and psutil:
            mem_mb = psutil.Process().memory_info().rss / (1024 ** 2)
            print(f"[CaptureThread] Queue={self.queue.qsize()}  Mem={mem_mb:.1f}MB")

    def get_frame(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None, None

"""
def wait_for_startup(capture_thread, delay_sec=3):
    start_time = time.time()
    while True:
        frame, _ = capture_thread.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        elapsed = time.time() - start_time
        remaining = max(0, delay_sec - elapsed) # noqa

        # Show "Initializing..." with countdown
        vis = frame.copy()
        cv2.putText(vis, f"Initializing... {remaining:.1f}s", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("ROI", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt

        # Once delay is done, flush buffer and return
        if elapsed >= delay_sec:
            with capture_thread.lock:
                capture_thread.deque.clear()
            return
"""

