# landmarker.py
import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerOptions
from config import needed_blendshapes

class FaceLandmarkerWrapper:
    def __init__(self):
        self.options   = self.get_face_landmarker_options()
        self.detector  = FaceLandmarker.create_from_options(self.options)
        self.connections =mp_tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION


    # ------------------------------------------------------------------
    def detect(self, frame_bgr,h,w, timestamp):
        timestamp_ms = int(timestamp * 1000)            # MP expects ms
        frame_rgb    = self.convert(frame_bgr)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if result and result.face_landmarks:
            lms = result.face_landmarks[0]
            raw_cords = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
            t_matrix = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)[:3, :3]
            blend = self.get_blendcoff(result.face_blendshapes,needed_blendshapes)
            return raw_cords, t_matrix, blend
        else:
            return None, None,None

    #convert to 720 p for processing
    @staticmethod
    def convert(frame):
        target_height = 720
        h, w = frame.shape[:2]
        if h > target_height:
            scale = target_height / h
            resized_frame = cv2.resize(frame, (int(w * scale), target_height))
        else:
            resized_frame = frame
        return cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    @staticmethod
    def get_blendcoff(blendshapes,keys):
        xd = [item for sublist in blendshapes for item in sublist]
        scores = {c.category_name: c.score for c in xd}
        return np.array([scores.get(k, 0.0) for k in keys], dtype=np.float32)

    @staticmethod
    def get_face_model_path():
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(base, "model", "face_landmarker.task"))

    def get_face_landmarker_options(self):
        base_options = mp_tasks.BaseOptions(
            model_asset_path=self.get_face_model_path(),
            delegate=mp_tasks.BaseOptions.Delegate.CPU
        )

        return FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=mp_tasks.vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )



