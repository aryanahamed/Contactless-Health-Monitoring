import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt

# --- Constants ---
MIN_HR_HZ = 0.67  # 40 bpm
MAX_HR_HZ = 3  # 180 bpm
MIN_PEAKS_FOR_HRV = 7 # Minimum peaks for HRV calculation

# --- Breathing Rate Constants ---
MIN_BR_HZ = 0.1  # 6 breaths per minute
MAX_BR_HZ = 0.5  # 30 breaths per minute
MIN_SIGNAL_LEN_FOR_BR = 60  # Samples for low freq analysis

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
        
        # height_threshold = np.mean(filtered_signal) + 0.1 * np.std(filtered_signal)
        prominence_threshold = np.std(filtered_signal) * 1
    
        peaks, _ = scipy.signal.find_peaks(
            filtered_signal,
            distance=min_distance_samples,
            prominence=prominence_threshold,
            height=None
        )
                
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
        # Calculate Inter Beat Intervals in seconds
        peak_diffs_samples = np.diff(peak_indices)
        ibis_sec = peak_diffs_samples / fps
        
        min_valid_ibi = 0.38  # minimum IBI in seconds (158 BPM)
        max_valid_ibi = 1.5   # maximum IBI in seconds (40 BPM)
        
        # First filter by physiological range
        valid_mask = (ibis_sec >= min_valid_ibi) & (ibis_sec <= max_valid_ibi)
        ibis_sec_filtered = ibis_sec[valid_mask]
        
        
        # If enough IBIs remain, perform statistical filtering
        if len(ibis_sec_filtered) >= MIN_PEAKS_FOR_HRV:
            
            median_ibi = np.median(ibis_sec_filtered)
            mad_ibi = scipy.stats.median_abs_deviation(ibis_sec_filtered, scale='normal')

            k_mad = 2.5
            lower_bound = median_ibi - k_mad * mad_ibi
            upper_bound = median_ibi + k_mad * mad_ibi

            # Ensure bounds stay within physiological limits
            lower_bound = max(lower_bound, min_valid_ibi)
            upper_bound = min(upper_bound, max_valid_ibi)

            ibis_sec_final = ibis_sec_filtered[(ibis_sec_filtered >= lower_bound) &
                                        (ibis_sec_filtered <= upper_bound)]
            
            
            if len(ibis_sec_final) < MIN_PEAKS_FOR_HRV:
                print(f"Not enough valid IBIs for HRV calculation after filtering. Only {len(ibis_sec_final)} valid IBIs.")
                return None
            
            if len(ibis_sec_final) >= MIN_PEAKS_FOR_HRV - 1 : # Check if enough IBIs remain
                mean_ibi_sec = np.mean(ibis_sec_final)
                hr_bpm = 60.0 / mean_ibi_sec
                sdnn_ms = np.std(ibis_sec_final) * 1000.0
                rmssd_ms = np.sqrt(np.mean(np.diff(ibis_sec_final) ** 2)) * 1000.0

                # MAX_ACCEPTABLE_SDNN = 150 # Example threshold in ms
                # if sdnn_ms > MAX_ACCEPTABLE_SDNN:
                #     print(f"HRV calculation abandoned: SDNN ({sdnn_ms:.1f}ms) exceeds stability threshold ({MAX_ACCEPTABLE_SDNN}ms).")
                #     return None
            
            # Check if HR and HRV values are in reasonable ranges
            if hr_bpm < 40 or hr_bpm > 158:
                print(f"Calculated HR {hr_bpm:.1f} BPM is outside physiological range.")
                return None
                
            if sdnn_ms > 500 or rmssd_ms > 500:
                print(f"Calculated HRV metrics are unrealistically high: SDNN={sdnn_ms:.1f}ms, RMSSD={rmssd_ms:.1f}ms")
                return None
            
            results = {'hr': hr_bpm, 'sdnn': sdnn_ms, 'rmssd': rmssd_ms}
            return results
        else:
            print(f"Not enough valid IBIs within physiological range. Only {len(ibis_sec_filtered)} valid IBIs.")
            return None
    else:
        print(f"Not enough peaks for HRV calculation. Need at least {MIN_PEAKS_FOR_HRV}, got {len(peak_indices) if peak_indices is not None else 0}.")
        return None


def extract_breathing_signal(rgb_buffer, fps):
    """
    Extracts and filters a potential breathing signal from the RGB data.
    Uses the detrended green channel as the primary source.

    Args:
        rgb_buffer (np.ndarray): A NumPy array of shape (N, 3) [R, G, B].
        fps (float): Frames per second.

    Returns:
        np.ndarray | None: The filtered breathing signal, or None if input is invalid.
    """
    if rgb_buffer is not None and rgb_buffer.shape[0] >= MIN_SIGNAL_LEN_FOR_BR and rgb_buffer.shape[1] == 3:
        # Select Green Channel
        green_signal = rgb_buffer[:, 1]

        # Detrend the signal
        detrended_green = scipy.signal.detrend(green_signal, type='linear')
        
        # Normalize the signal
        mean_green = np.mean(detrended_green)
        std_green = np.std(detrended_green)
        if std_green == 0:
            return None
        detrended_green = (detrended_green - mean_green) / std_green
        
        # 3. Filter for Breathing Rate frequencies
        from pos_processing import apply_butterworth_bandpass
        try:
            filtered_br_signal = apply_butterworth_bandpass(
                detrended_green, MIN_BR_HZ, MAX_BR_HZ, fps, order=2  # Lower order better for BR
            )
            return filtered_br_signal
        except ValueError as e:
            print(f"Error filtering breathing signal: {e}")
            return None
    else:
        if rgb_buffer is None:
            print("Invalid RGB buffer provided for BR extraction.")
        elif rgb_buffer.shape[0] < MIN_SIGNAL_LEN_FOR_BR:
            print(f"RGB buffer too short for BR extraction. Need {MIN_SIGNAL_LEN_FOR_BR}, got {rgb_buffer.shape[0]}.")
        else:
            print("Invalid RGB buffer shape for BR extraction.")
        return None


def calculate_breathing_rate_welch(breathing_signal, fps):
    """
    Calculates the Breathing Rate (BR) using Welch's method to estimate power spectrum.
    
    Args:
        breathing_signal (np.ndarray): The filtered 1D breathing signal.
        fps (float): The frame rate (samples per second) of the signal.

    Returns:
        float | None: Estimated breathing rate in Breaths Per Minute (BPM),
                      or None if unable to determine.
    """
    if breathing_signal is None or len(breathing_signal) < MIN_SIGNAL_LEN_FOR_BR:
        print(f"Insufficient data for BR calculation. Need at least {MIN_SIGNAL_LEN_FOR_BR} samples.")
        return None

    # Power Spectral Density
    freqs, power = scipy.signal.welch(
        breathing_signal, fs=fps, nperseg=min(256, len(breathing_signal))
    )

    # Limit breathing frequency band
    br_band_indices = np.where((freqs >= MIN_BR_HZ) & (freqs <= MAX_BR_HZ))[0]

    if len(br_band_indices) == 0:
        print("No frequency components found within the defined BR band (Welch).")
        return None

    peak_index_in_band = np.argmax(power[br_band_indices])
    peak_global_index = br_band_indices[peak_index_in_band]
    dominant_freq_hz = freqs[peak_global_index]
    breathing_rate_bpm = dominant_freq_hz * 60.0

    if breathing_rate_bpm < (MIN_BR_HZ * 60) or breathing_rate_bpm > (MAX_BR_HZ * 60):
        print(f"Calculated BR {breathing_rate_bpm:.1f} BPM is outside expected range.")
        return None

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, power, 'b-')
    plt.axvline(dominant_freq_hz, color='r', linestyle='--', 
                label=f'Breathing Rate: {breathing_rate_bpm:.1f} BPM')
    plt.axvspan(MIN_BR_HZ, MAX_BR_HZ, alpha=0.2, color='green', 
                label=f'Breathing Band ({MIN_BR_HZ*60:.1f}-{MAX_BR_HZ*60:.1f} BPM)')
    plt.title('Breathing Signal Frequency Spectrum (Welch)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.xlim(0, MAX_BR_HZ * 2)
    plt.legend()
    plt.grid(True)
    plt.savefig('breathing_spectrum_welch.png')
    plt.close()

    return breathing_rate_bpm