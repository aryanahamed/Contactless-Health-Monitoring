import numpy as np
import scipy.signal
import scipy.stats
from config import (
    MAX_HR_HZ,
    MIN_PEAKS_FOR_HRV,
    MIN_VALID_IBI_S,
    MAX_VALID_IBI_S,
    MAX_ACCEPTABLE_SDNN_MS,
    MAX_ACCEPTABLE_RMSSD_MS,
    MIN_BR_HZ,
    MAX_BR_HZ,
    MIN_SAMPLES_FOR_BR,
    LOMB_SCARGLE_FREQ_POINTS,
)

from pos_processing import apply_butterworth_bandpass

def find_signal_peaks(filtered_resampled_signal, target_fps):
    if filtered_resampled_signal is None or len(filtered_resampled_signal) == 0 or target_fps <= 0:
        print("Invalid signal or FPS for peak finding.")
        return None

    min_distance_samples = max(1, int(target_fps / MAX_HR_HZ))
    signal_std = np.std(filtered_resampled_signal)
    if signal_std == 0:
        print("Signal standard deviation is zero, cannot determine prominence.")
        return None
    
    prominence_threshold = signal_std * 0.4
    height_threshold = None

    try:
        peaks = scipy.signal.find_peaks(
            filtered_resampled_signal,
            distance=min_distance_samples,
            prominence=prominence_threshold,
            height=height_threshold
        )

        if len(peaks[0]) > 0:
            print(f"Found {len(peaks[0])} peaks in resampled signal.")
            return peaks
        else:
            print("No peaks found in the filtered resampled signal.")
            return None
    except Exception as e:
        print(f"Error during peak finding: {e}")
        return None


def map_resampled_peaks_to_original_timestamps(resampled_peak_indices, uniform_timestamps, original_timestamps):
    if resampled_peak_indices is None or uniform_timestamps is None or original_timestamps is None:
        print("Error: Invalid input for peak mapping.")
        return None
    if len(resampled_peak_indices) == 0:
        print("No resampled peaks to map.")
        return np.array([])

    try:
        # Get the time values of the peaks found on the uniform grid
        peak_times_uniform = uniform_timestamps[resampled_peak_indices]
        
        # Ensure original_timestamps are sorted
        if not np.all(np.diff(original_timestamps) >= 0):
            print("Warning: Original timestamps not sorted for mapping. Sorting now.")
            original_timestamps = np.sort(original_timestamps)

        # Find the closest original timestamps to the resampled peak times
        closest_orig_indices = np.searchsorted(original_timestamps, peak_times_uniform, side='left')
        closest_orig_indices = np.clip(closest_orig_indices, 0, len(original_timestamps) - 1)

        # Handle edge cases where the peak time is before the first original timestamp
        left_indices = closest_orig_indices - 1
        left_indices[left_indices < 0] = 0
        
        # Calculate distances to determine which original timestamp to use
        dist_curr = np.abs(original_timestamps[closest_orig_indices] - peak_times_uniform)
        dist_left = np.abs(original_timestamps[left_indices] - peak_times_uniform)

        # Choose the closest original timestamp
        use_left_mask = (dist_left < dist_curr) & (closest_orig_indices > 0)
        final_orig_indices = np.where(use_left_mask, left_indices, closest_orig_indices)
        
        # Extract the timestamp values
        mapped_peak_timestamps = original_timestamps[final_orig_indices]

        # Remove potential duplicate timestamps
        unique_mapped_timestamps = np.unique(mapped_peak_timestamps)

        print(f"Mapped {len(resampled_peak_indices)} resampled peaks to {len(unique_mapped_timestamps)} unique original timestamps.")
        return unique_mapped_timestamps

    except Exception as e:
        print(f"Error mapping peaks to original timestamps: {e}")
        return None


def calculate_hr_hrv(peak_timestamps):
    if peak_timestamps is None or len(peak_timestamps) < MIN_PEAKS_FOR_HRV:
        print(f"Not enough peak timestamps for HRV. Need at least {MIN_PEAKS_FOR_HRV}, got {len(peak_timestamps) if peak_timestamps is not None else 0}.")
        return None

    # Calculate Inter Beat Intervals directly in seconds from timestamps
    ibis_sec = np.diff(peak_timestamps)

    if len(ibis_sec) < MIN_PEAKS_FOR_HRV -1 : # Need at least N-1 IBIs for N peaks
        print(f"Not enough IBIs ({len(ibis_sec)}) derived from peaks for HRV.")
        return None

    # Filter by physiological range
    valid_mask_physio = (ibis_sec >= MIN_VALID_IBI_S) & (ibis_sec <= MAX_VALID_IBI_S)
    ibis_sec_physio = ibis_sec[valid_mask_physio]

    if len(ibis_sec_physio) < MIN_PEAKS_FOR_HRV - 1:
        print(f"Not enough IBIs ({len(ibis_sec_physio)}) within physiological range ({MIN_VALID_IBI_S*1000:.0f}-{MAX_VALID_IBI_S*1000:.0f} ms).")
        return None

    # Statistical filtering
    median_ibi = np.median(ibis_sec_physio)
    mad_ibi = scipy.stats.median_abs_deviation(ibis_sec_physio, scale='normal', nan_policy='omit')

    if np.isnan(mad_ibi) or mad_ibi == 0:
        print("Warning: MAD calculation resulted in zero or NaN. Using only physiological filter.")
        ibis_sec_final = ibis_sec_physio
    else:
        k_mad = 2.5
        lower_bound = median_ibi - k_mad * mad_ibi
        upper_bound = median_ibi + k_mad * mad_ibi

        # Ensure bounds stay within limits after
        lower_bound = max(lower_bound, MIN_VALID_IBI_S)
        upper_bound = min(upper_bound, MAX_VALID_IBI_S)

        valid_mask_stat = (ibis_sec_physio >= lower_bound) & (ibis_sec_physio <= upper_bound)
        ibis_sec_final = ibis_sec_physio[valid_mask_stat]

    if len(ibis_sec_final) < MIN_PEAKS_FOR_HRV - 1:
        print(f"Not enough IBIs ({len(ibis_sec_final)}) remaining after statistical filtering.")
        return None

    print(f"Final number of IBIs used for HR/HRV: {len(ibis_sec_final)}")

    mean_ibi_sec = np.mean(ibis_sec_final)
    if mean_ibi_sec <= 0: return None

    hr_bpm = 60.0 / mean_ibi_sec
    sdnn_ms = np.std(ibis_sec_final) * 1000.0
    if len(ibis_sec_final) >= 2:
        rmssd_ms = np.sqrt(np.mean(np.diff(ibis_sec_final) ** 2)) * 1000.0
    else:
        rmssd_ms = np.nan

    if hr_bpm < (60.0/MAX_VALID_IBI_S) or hr_bpm > (60.0/MIN_VALID_IBI_S):
        print(f"Warning: Calculated HR {hr_bpm:.1f} BPM is outside implied physiological range.")

    # Check HRV metrics stability
    if sdnn_ms > MAX_ACCEPTABLE_SDNN_MS:
        print(f"SDNN ({sdnn_ms:.1f}ms) exceeds stability threshold ({MAX_ACCEPTABLE_SDNN_MS}ms).")
        return {'hr': hr_bpm, 'sdnn': np.nan, 'rmssd': np.nan, 'hrv_quality': 'unstable_sdnn'}
    if not np.isnan(rmssd_ms) and rmssd_ms > MAX_ACCEPTABLE_RMSSD_MS:
        print(f"RMSSD ({rmssd_ms:.1f}ms) exceeds stability threshold ({MAX_ACCEPTABLE_RMSSD_MS}ms).")
        return {'hr': hr_bpm, 'sdnn': sdnn_ms, 'rmssd': np.nan, 'hrv_quality': 'unstable_rmssd'}


    results = {'hr': hr_bpm, 'sdnn': sdnn_ms, 'rmssd': rmssd_ms, 'hrv_quality': 'ok'}
    print(f"Calculated HR: {hr_bpm:.1f} BPM, SDNN: {sdnn_ms:.1f} ms, RMSSD: {rmssd_ms:.1f} ms")
    return results


def extract_breathing_signal(rgb_buffer, timestamps):
    # Handle NaN values
    from pos_processing import handle_nan_values
    cleaned_rgb, cleaned_timestamps = handle_nan_values(rgb_buffer, timestamps)

    if cleaned_rgb is None or cleaned_timestamps is None:
        print("BR extraction failed: Invalid data after NaN handling.")
        return None, None

    if cleaned_rgb.shape[0] < MIN_SAMPLES_FOR_BR:
        print(f'''BR extraction failed: Need {MIN_SAMPLES_FOR_BR} samples after NaN removal,
            got {cleaned_rgb.shape[0]}.''')
        return None, None
    
    if len(cleaned_timestamps) > 1:
        approx_fps = 1.0 / np.mean(np.diff(cleaned_timestamps))
    else:
        approx_fps = 1.0

    # Select Green Channel
    green_signal = cleaned_rgb[:, 1]

    # Detrend
    detrended_green = scipy.signal.detrend(green_signal, type='linear')
    
    filtered_breathing_signal = apply_butterworth_bandpass(
        detrended_green, MIN_BR_HZ, MAX_BR_HZ, approx_fps, order=2 
    )
    
    if filtered_breathing_signal is None:
        print("BR extraction failed: Filtering step failed.")
        return None, None

    # Normalize
    mean_green = np.mean(filtered_breathing_signal)
    std_green = np.std(filtered_breathing_signal)
    if std_green == 0:
        print("BR extraction failed: Filtered green channel standard deviation is zero.")
        return None, None
    
    detrended_normalized_green = (filtered_breathing_signal - mean_green) / std_green

    print(f"Breathing signal extracted and filtered using {len(cleaned_timestamps)} samples.")
    return detrended_normalized_green, cleaned_timestamps


def calculate_breathing_rate_lombscargle(breathing_signal, timestamps, plot_spectrum=True):
    if breathing_signal is None or timestamps is None:
        print("BR calculation failed: Invalid input signal or timestamps.")
        return None
    if len(breathing_signal) != len(timestamps) or len(breathing_signal) < MIN_SAMPLES_FOR_BR:
        print(f"BR calculation failed: Insufficient data ({len(breathing_signal)}) or mismatched lengths.")
        return None

    frequencies_hz = np.linspace(MIN_BR_HZ * 0.8, MAX_BR_HZ * 1.2, LOMB_SCARGLE_FREQ_POINTS)

    try:
        # Calculate Lomb-Scargle Power Spectrum
        power = scipy.signal.lombscargle(timestamps, breathing_signal, frequencies_hz, normalize=True)

        # Find the peak power within the BR band
        br_band_mask = (frequencies_hz >= MIN_BR_HZ) & (frequencies_hz <= MAX_BR_HZ)
        if not np.any(br_band_mask):
            print("BR calculation failed: No frequencies evaluated within the defined BR band.")
            return None

        power_in_band = power[br_band_mask]
        freqs_in_band = frequencies_hz[br_band_mask]

        if len(power_in_band) == 0:
            print("BR calculation failed: Error filtering frequencies in the BR band.")
            return None

        peak_index_in_band = np.argmax(power_in_band)
        peak_power = power_in_band[peak_index_in_band]
        dominant_freq_hz = freqs_in_band[peak_index_in_band]
        breathing_rate_bpm = dominant_freq_hz * 60.0
        
        # Add peak power to log
        print(f"Calculated Breathing Rate: {breathing_rate_bpm:.1f} BPM (from {dominant_freq_hz:.3f} Hz, Power: {peak_power:.4f})")     
        print(f"Calculated Breathing Rate: {breathing_rate_bpm:.1f} BPM (from {dominant_freq_hz:.3f} Hz)")

        if breathing_rate_bpm < (MIN_BR_HZ * 60 * 0.9) or breathing_rate_bpm > (MAX_BR_HZ * 60 * 1.1):
            print(f"Calculated BR {breathing_rate_bpm:.1f} BPM is outside expected range after peak finding.")
        return breathing_rate_bpm

    except Exception as e:
        print(f"Error during Lomb-Scargle calculation for BR: {e}")
        import traceback
        traceback.print_exc()
        return None