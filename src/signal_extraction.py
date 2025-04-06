import numpy as np
import scipy.signal

# --- Constants ---
MIN_HR_HZ = 0.67  # 40 bpm
MAX_HR_HZ = 3.33  # 200 bpm
MIN_PEAKS_FOR_HRV = 5

def find_signal_peaks(filtered_signal, fps):
    """
    Finds peaks in the filtered physiological signal corresponding to heartbeats.

    Args:
        filtered_signal (np.ndarray): The 1D filtered pulse signal (should be clean).
        fps (float): The frame rate (samples per second) of the signal.

    Returns:
        np.ndarray | None: An array containing the indices (sample number) of the
                           detected peaks. Returns None if the signal is invalid
                           or no suitable peaks are found.
    """
    if filtered_signal is not None and len(filtered_signal) > 0:
        # Minimum peak distance based on maximum expected HR
        min_distance_samples = int(fps / MAX_HR_HZ)
        min_distance_samples = max(min_distance_samples, 1)
        
        # Find peaks
        peaks, _ = scipy.signal.find_peaks(filtered_signal, distance=min_distance_samples, height=0.1, prominence=0.7)
        if len(peaks) > 0:
            return peaks
        else:
            print("No peaks found in the filtered signal.")
            return None
    else:
        print("Invalid filtered signal provided.")
        return None


def calculate_hr_hrv(peak_indices, fps):
    """
    Calculates Heart Rate (HR) and time-domain HRV metrics (SDNN, RMSSD)
    from peak timings.

    Args:
        peak_indices (np.ndarray): Array of detected peak indices (sample numbers).
        fps (float): The frame rate (samples per second) used when acquiring the signal.

    Returns:
        dict | None: A dictionary containing calculated metrics:
                      {'hr': float,      # Heart rate in Beats Per Minute (BPM)
                       'sdnn': float,    # SDNN in milliseconds (ms)
                       'rmssd': float}   # RMSSD in milliseconds (ms)
                    Returns None if not enough peaks are provided for reliable calculation.
    """
    if peak_indices is not None and len(peak_indices) >= MIN_PEAKS_FOR_HRV:
        # Calculate Inter-Beat Intervals (IBIs) in seconds
        peak_diffs_samples = np.diff(peak_indices)
        ibis_sec = peak_diffs_samples / fps
        
        # Outlier Removal for IBIs
        mean_ibi = np.mean(ibis_sec)
        std_ibi = np.std(ibis_sec)
        lower_bound = mean_ibi - 2 * std_ibi
        upper_bound = mean_ibi + 2 * std_ibi
        ibis_sec = ibis_sec[(ibis_sec >= lower_bound) & (ibis_sec <= upper_bound)]
        if len(ibis_sec) < MIN_PEAKS_FOR_HRV:
            print("Not enough valid IBIs for HRV calculation.")
            return None
        
        # Calculate Heart Rate (HR) in BPM
        mean_ibi_sec = np.mean(ibis_sec)
        if mean_ibi_sec <= 0:
            print("Mean IBI is zero or negative, cannot calculate HR.")
            return None
        
        hr_bpm = 60.0 / mean_ibi_sec
        
        # Calculate SDNN in milliseconds
        sdnn_ms = np.std(ibis_sec) * 1000.0
        
        # Calculate RMSSD in milliseconds
        rmmsd_ms = np.sqrt(np.mean(np.diff(ibis_sec) ** 2)) * 1000.0
        
        results = {'hr': hr_bpm, 'sdnn': sdnn_ms, 'rmssd': rmmsd_ms}
        return results
    else:
        print("Not enough peaks for HRV calculation.")
        return None