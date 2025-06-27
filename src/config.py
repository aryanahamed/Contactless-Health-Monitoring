# config.py

# here we gonna store all the global pipeline parameters and some global functions
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerOptions
import os
import numpy as np



# camera settings
# camera_id = "src/vid.avi" # depends on ur device figure it out
camera_id = 0  # default camera ID for webcam


# landmark indices for each region
roi_landmarks = {
    "forehead": np.array([109,108,107,55,8,285,336,337,338,10,151,9]),
    "left_cheek": np.array([117,118,119,120,100,142,36,205,50,101]),
    "right_cheek": np.array([349,348,347,346,280,425,266,371,329,330]),
}


# buffer settings for time series
window = 12
hz = 30
buffer_size = hz * window

# regions we care about for rPPG
regions = ["forehead", "left_cheek", "right_cheek"]

# returns the absolute path to the face_landmarker.task model
def get_face_model_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, "ROI/model", "face_landmarker.task"))


# mediapipe face landmarker settings
def get_face_landmarker_options():
        base_options = mp_tasks.BaseOptions(
            model_asset_path=get_face_model_path(),
            delegate=mp_tasks.BaseOptions.Delegate.CPU
        )

        return FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=mp_tasks.vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )



# -- Constants -- #

# -- pos_processing -- #
BAND_MIN_HZ = 0.67  # 40 BPM
BAND_MAX_HZ = 3.0  # 180 BPM
MIN_SAMPLES_FOR_POS = 340
MIN_SAMPLES_FOR_QUALITY = 340
DEFAULT_TARGET_FPS = 30.0


# -- signal_extraction -- #
MIN_HR_HZ = 0.67  # 40 bpm
MAX_HR_HZ = 3.0  # 180 bpm
MIN_PEAKS_FOR_HRV = 12 # Minimum peaks for HRV calculation
MIN_VALID_IBI_S = 0.333  # Min IBI in seconds (180 BPM)
MAX_VALID_IBI_S = 1.5  # Max IBI in seconds (40 BPM)
MAX_ACCEPTABLE_SDNN_MS = 200
MAX_ACCEPTABLE_RMSSD_MS = 200

MIN_BR_HZ = 0.1  # 6 breaths per minute
MAX_BR_HZ = 0.5  # 30 breaths per minute
MIN_SAMPLES_FOR_BR = 340
RESP_SIG_INTERPOLATION_FS = 4.0 # Hz for interpolation of BR


# Smoothing
SMOOTHING_WINDOW_SIZE = 5


