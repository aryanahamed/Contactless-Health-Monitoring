# config.py

# here we gonna store all the global pipeline parameters and some global functions

import os
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerOptions


# camera settings
camera_id = 0 # depends on ur device figure it out
GROUND_TRUTH = "test_inputs/ground_truth.txt"
frame_width = 640 #640
frame_height = 480 #480

# landmark indices for each region
roi_landmarks = {
    "forehead": np.array([109,108,107,55,8,285,336,337,338,10,151,9]),
    "left_cheek": np.array([117,118,119,142,36,205,50,101]),
    "right_cheek": np.array([348,347,346,280,425,266,329]),
}


# buffer settings for time series
window = 12
buffer_size = 30 * window

# regions we care about for rPPG
regions = ["forehead", "left_cheek", "right_cheek"]

# returns the absolute path to the face_landmarker.task model
def get_face_model_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, "ROI/model", "face_landmarker.task"))

def get_segmentation_model_path():
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base, "ROI/model", "segment_multi.tflite"))


# mediapipe face landmarker settings
def get_face_landmarker_options():
    try:
        base_options = mp_tasks.BaseOptions(
            model_asset_path=get_face_model_path(),
            delegate=mp_tasks.BaseOptions.Delegate.CPU
            #delegate=mp_tasks.BaseOptions.Delegate.GPU
        )
    except Exception as e:
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
        min_face_detection_confidence=0.8,
        min_face_presence_confidence=0.8,
        min_tracking_confidence=0.7
    )


# -- Constants -- #

# -- pos_processing -- #
BAND_MIN_HZ = 0.67  # 40 BPM
BAND_MAX_HZ = 3.33   # 200 BPM
MIN_SAMPLES_FOR_POS = 300
MIN_SAMPLES_FOR_QUALITY = 300
DEFAULT_TARGET_FPS = 30.0


# -- signal_extraction -- #
MAX_HR_HZ = 3.33  # 200 bpm
MIN_PEAKS_FOR_HRV = 12
MIN_VALID_IBI_S = 0.3  # Min IBI in seconds (200 BPM)
MAX_VALID_IBI_S = 1.5  # Max IBI in seconds (40 BPM)
MAX_ACCEPTABLE_SDNN_MS = 300
MAX_ACCEPTABLE_RMSSD_MS = 300

MIN_BR_HZ = 0.1  # 6 breaths per minute
MAX_BR_HZ = 0.5  # 30 breaths per minute
MIN_SAMPLES_FOR_BR = 300


