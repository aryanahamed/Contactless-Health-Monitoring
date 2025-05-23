import threading
import collections
import cv2
from config import camera_id
import time
import os
import psutil
#sets up a thread for capturing and using a dequeue to store the frame
#and allowing a get method for the main to get the frame
class CaptureThread:
    # not sure if to not limit queue size, might cause series memory issues
    def __init__(self, max_queue_size=2000):
        #initialize the thread
        self.cap = cv2.VideoCapture(camera_id)
        # checking if camera id is a video file not webcam
        self.use_video = isinstance(camera_id, str)
        self.deque = collections.deque(maxlen=max_queue_size) #dequeue to store frmaes with timestamps
        self.running = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True) #daemon true just allows main to exit
        self.lock = threading.Lock() #lock to prevent data inconsistency
        # only calculating fps if it's a video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.use_video else None
        self.frame_idx = 0

    #self-explanatory
    def start(self):
        self.running.set()
        self.thread.start()

    def stop(self):
        self.running.clear()
        self.thread.join()
        self.cap.release()

    def _capture_loop(self):
        while self.running.is_set():
            success, frame = self.cap.read()
            frame = cv2.flip(frame, 1) #flipping on axis y
            if not success:
                time.sleep(0.01) # sleep for a fit to avoid busy waiting
                continue
            if self.use_video:
                # calculate the timestamp for video
                # checking if the video fps is valid, if not use 30
                fps = self.fps if self.fps and self.fps > 1e-3 else 30
                timestamp = self.frame_idx / fps
                self.frame_idx += 1
            else:
                timestamp = time.perf_counter()
            with self.lock:
                self.deque.append((frame, timestamp))
            # print(f"[CaptureThread] Queue size: {len(self.deque)}")
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 ** 2)
            # print(f"[Memory] After deque append - {mem_mb:.2f} MB")

    def get_frame(self):
        with self.lock:
            if self.deque:
                return self.deque.popleft() # taking the first value and removing it so to not double read
        return None, None



def wait_for_startup(capture_thread, delay_sec=3):
    start_time = time.time()

    while True:
        frame, _ = capture_thread.get_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        elapsed = time.time() - start_time
        remaining = max(0, delay_sec - elapsed)

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
            # print(f"[Startup] Delay complete after {elapsed:.2f}s. Buffer flushed.")
            return