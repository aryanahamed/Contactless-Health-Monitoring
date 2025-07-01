# config.py
import numpy as np
# here we gonna store all the global pipeline parameters and some global functions


# landmark indices for each region
roi_indices = {
    "forehead": np.array([109,10,338,337,336,9,107,108,151]),
    "left_cheek": np.array([117,118,119,120,100,142,36,205,187,123,101,50]),
    "right_cheek": np.array([349,348,347,346,352,411,280,425,266,371,329,330])
}


anchor_indices = np.array([1, 4, 5,195,197])

roi_idx = {67,109,10,338,297,299,336,9,107,69,67,108,151,337,
           117,118,119,120,100,142,36,205,187,123,101,50,
           349,348,347,346,352,411,280,425,266,371,329,330}


# buffer settings for time series
window = 15
hz = 30
buffer_size = (hz * window)

# regions we care about for rPPG
regions = ["forehead", "left_cheek", "right_cheek"]

#sorted index wise
needed_blendshapes = [
    "browDownLeft",        # 0 - stress
    "browDownRight",       # 1 - stress
    "browInnerUp",         # 2 - stress/concentration + forehead ROI
    "browOuterUpLeft",     # 3 - forehead ROI
    "browOuterUpRight",    # 4 - forehead ROI
    "cheekPuff",           # 5 - cheek ROI
    "cheekSquintLeft",     # 6 - cheek ROI
    "cheekSquintRight",    # 7 - cheek ROI
    "eyeBlinkLeft",        # 8 - blink
    "eyeBlinkRight",       # 9 - blink
    "eyeLookDownLeft",     # 10 - eye gaze vertical
    "eyeLookDownRight",    # 11 - eye gaze vertical
    "eyeLookInLeft",       # 12 - eye gaze horizontal
    "eyeLookInRight",      # 13 - eye gaze horizontal
    "eyeLookOutLeft",      # 14 - eye gaze horizontal
    "eyeLookOutRight",     # 15 - eye gaze horizontal
    "eyeLookUpLeft",       # 16 - eye gaze vertical
    "eyeLookUpRight",      # 17 - eye gaze vertical
    "eyeSquintLeft",       # 18 - stress/eye opening
    "eyeSquintRight",      # 19 - stress/eye opening
    "eyeWideLeft",         # 20 - attention/eye opening
    "eyeWideRight",        # 21 - attention/eye opening
    "mouthSmileLeft",      # 22 - stress/mood + cheek ROI
    "mouthSmileRight"      # 23 - stress/mood + cheek ROI
]


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






