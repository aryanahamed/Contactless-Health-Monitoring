# landmarker.py

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import FaceLandmarker
from config import get_face_landmarker_options

class FaceLandmarkerWrapper:
    def __init__(self):
        self.options = get_face_landmarker_options()
        self.detector = FaceLandmarker.create_from_options(self.options)

    def detect(self, frame_bgr, timestamp):
        """
        returns facial_landmarks[0] or None if no face is found.
        """
        timestamp_ms = int(timestamp * 1000) # mp needs time in ms
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        #create the mp object image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        #storing the landmark detection result
        result = self.detector.detect_for_video(mp_image,timestamp_ms)

        if result and result.face_landmarks:
            #returning landmarks and mp image(for segmentation, t_matrix for angle)
            t_matrix = result.facial_transformation_matrixes[0].data
            return result.face_landmarks[0],t_matrix

        else:
            return None,None

