# -- Constants -- #

# -- pos_processing -- #
BAND_MIN_HZ = 0.75  # 45 BPM
BAND_MAX_HZ = 3.0   # 180 BPM 
MIN_SAMPLES_FOR_POS = 60
MIN_SAMPLES_FOR_RESAMPLE = 60
MIN_SAMPLES_FOR_QUALITY = 64
DEFAULT_TARGET_FPS = 30.0


# -- signal_extraction -- #
MAX_HR_HZ = 3.0  # 180 bpm
MIN_PEAKS_FOR_HRV = 30
MIN_VALID_IBI_S = 0.38  # Min IBI in seconds (158 BPM)
MAX_VALID_IBI_S = 1.5   # Max IBI in seconds (40 BPM)
MAX_ACCEPTABLE_SDNN_MS = 200
MAX_ACCEPTABLE_RMSSD_MS = 200

MIN_BR_HZ = 0.1  # 6 breaths per minute
MAX_BR_HZ = 0.4  # 24 breaths per minute
MIN_SAMPLES_FOR_BR = 60
LOMB_SCARGLE_FREQ_POINTS = 500