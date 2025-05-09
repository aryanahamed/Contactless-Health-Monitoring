import numpy as np
import scipy.signal
import scipy.interpolate
from config import (
    MIN_SAMPLES_FOR_POS,
    MIN_SAMPLES_FOR_RESAMPLE,
    MIN_SAMPLES_FOR_QUALITY,
    BAND_MIN_HZ,
    BAND_MAX_HZ,
    DEFAULT_TARGET_FPS
)

# import cv2
# def extract_average_rgb(roi_frame):
#   if roi_frame is not None and roi_frame.ndim == 3 and roi_frame.shape[2] == 3:
#     mean_channels = np.mean(roi_frame, axis=(0, 1))
#     # Convert BGR mean to RGB
#     mean_rgb = (mean_channels[2], mean_channels[1], mean_channels[0])
#     return mean_rgb
#   else:
#     return None

def handle_nan_values(rgb_data, timestamps):
    if rgb_data is None or timestamps is None or rgb_data.shape[0] != timestamps.shape[0]:
        print("Error: Invalid input for NaN handling.")
        return None, None

    nan_mask = ~np.isnan(rgb_data).any(axis=1)
    cleaned_rgb = rgb_data[nan_mask]
    cleaned_timestamps = timestamps[nan_mask]

    if cleaned_rgb.shape[0] < MIN_SAMPLES_FOR_POS:
        print(f"Warning: Not enough valid data points ({cleaned_rgb.shape[0]}) after NaN removal.")
        return None, None

    if not np.all(np.diff(cleaned_timestamps) > 0):
        print("Warning: Timestamps are not strictly increasing after NaN removal. Attempting to fix.")
        unique_indices = np.unique(cleaned_timestamps, return_index=True)[1]
        unique_indices = np.sort(unique_indices)
        cleaned_timestamps = cleaned_timestamps[unique_indices]
        cleaned_rgb = cleaned_rgb[unique_indices]
        if cleaned_rgb.shape[0] < MIN_SAMPLES_FOR_POS or not np.all(np.diff(cleaned_timestamps) > 0):
            print("Error: Cannot fix non-monotonic timestamps or insufficient data after fixing.")
            return None, None

    return cleaned_rgb, cleaned_timestamps

def apply_pos_projection(rgb_buffer):
    if rgb_buffer is None or rgb_buffer.shape[0] < MIN_SAMPLES_FOR_POS:
        print(f'''POS requires at least {MIN_SAMPLES_FOR_POS} samples, 
            got {rgb_buffer.shape[0] if rgb_buffer is not None else 0}''')
        return None

    # Normalize by mean per channel
    mean_rgb = np.mean(rgb_buffer, axis=0)
    if np.any(mean_rgb == 0):
        print("Warning: Mean RGB contains zero, potential division by zero in POS.")
        return None
    normalized_rgb = rgb_buffer / mean_rgb

    # Detrend
    detrended_rgb = scipy.signal.detrend(normalized_rgb, axis=0)

    # Apply projection matrix
    X = detrended_rgb[:, 1] - detrended_rgb[:, 2] # G - B
    Y = detrended_rgb[:, 1] + detrended_rgb[:, 2] - 2*detrended_rgb[:, 0]  # G + B - 2R

    std_Y = np.std(Y)
    if std_Y == 0:
        print("SDV_Y ZERO")
        return None

    alpha = np.std(X) / std_Y
    S = X - alpha * Y

    return S

def resample_signal(signal, original_timestamps, target_fps=None):
    if signal is None or original_timestamps is None or len(signal) != len(original_timestamps):
        print("Error: Invalid input for resampling.")
        return None, None, None
    if len(signal) < MIN_SAMPLES_FOR_RESAMPLE:
        print(f"Error: Not enough data points ({len(signal)}) for resampling.")
        return None, None, None

    # Ensure timestamps are sorted
    sort_indices = np.argsort(original_timestamps)
    original_timestamps = original_timestamps[sort_indices]
    signal = signal[sort_indices]

    duration = original_timestamps[-1] - original_timestamps[0]
    if duration <= 0:
        print(f"Warning: Signal duration is non-positive ({duration}). Cannot resample.")
        return None, None, None

    if target_fps is None:
        num_samples = len(original_timestamps)
        estimated_fps = (num_samples - 1) / duration
        actual_target_fps = round(estimated_fps)
        if actual_target_fps <= 0: actual_target_fps = DEFAULT_TARGET_FPS
        print(f"Resampling: Estimated FPS = {estimated_fps:.2f}, Target FPS = {actual_target_fps}")
    else:
        actual_target_fps = target_fps

    num_uniform_samples = int(duration * actual_target_fps) + 1
    if num_uniform_samples < 2:
        print(f"Warning: Too few samples ({num_uniform_samples}) requested for resampling.")
        return None, None, None

    uniform_timestamps = np.linspace(original_timestamps[0], original_timestamps[-1], num_uniform_samples)

    try:
        interp_func = scipy.interpolate.interp1d(
            original_timestamps,
            signal,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        resampled_signal = interp_func(uniform_timestamps)
        return resampled_signal, uniform_timestamps, actual_target_fps

    except ValueError as e:
        print(f"Error during interpolation: {e}")
        return None, None, None

def apply_butterworth_bandpass(signal_buffer, low_cut_hz, high_cut_hz, fps, order=5):
    if signal_buffer is None or fps <= 0:
        return None
    if len(signal_buffer) <= order * 3:
        print(f"Warning: Signal length ({len(signal_buffer)}) too short for filter order ({order}).")
        return None

    nyquist = 0.5 * fps
    low = low_cut_hz / nyquist
    high = high_cut_hz / nyquist

    if low <= 0:
        print(f"Warning: Low cut frequency ({low_cut_hz} Hz) is too low, adjusting to small positive value.")
        low = 0.01 # Avoiding zero frequency
    if high >= 1:
        print(f"Warning: High cut frequency ({high_cut_hz} Hz) is >= Nyquist Freq ({nyquist} Hz). Clamping.")
        high = 0.99

    if low >= high:
        print(f"Error: Low cut frequency ({low*nyquist:.2f} Hz) is >= high cut frequency ({high*nyquist:.2f} Hz). Cannot create bandpass filter.")
        return None

    try:
        b, a = scipy.signal.butter(order, [low, high], btype='band')
        filtered_signal = scipy.signal.filtfilt(b, a, signal_buffer)
        return filtered_signal
    except ValueError as e:
        print(f"Error applying Butterworth filter: {e}")
        return None

def calculate_signal_quality(filtered_resampled_signal, fps):
    if filtered_resampled_signal is None or len(filtered_resampled_signal) < MIN_SAMPLES_FOR_QUALITY or fps <= 0:
        return 0.0
    try:
        # Power Spectral Density on the uniformly sampled signal
        freqs, power_spectrum = scipy.signal.welch(
            filtered_resampled_signal, fs=fps, nperseg=min(256, len(filtered_resampled_signal))
        )

        # Power in the desired heart rate band
        band_mask = (freqs >= BAND_MIN_HZ) & (freqs <= BAND_MAX_HZ)
        if not np.any(band_mask):
            print("Warning: No frequency components found in the target HR band for quality assessment.")
            return 0.0

        power_in_band = np.sum(power_spectrum[band_mask])

        # Total power in a broader reasonable physiological band
        total_mask = (freqs >= 0.5) & (freqs <= 10.0)
        if not np.any(total_mask):
            print("Warning: No frequency components found in the total power band for quality assessment.")
            return 0.0

        total_power = np.sum(power_spectrum[total_mask])
        
        if total_power <= 1e-10:
            print("Warning: Total power is near zero. Setting quality to 0.")
            return 0.0
        
        # Ensuring power_in_band does not exceed total_power
        power_in_band = min(power_in_band, total_power) 
        noise_power = total_power - power_in_band
        
        # Use a small epsilon to avoid division by zero
        epsilon = 1e-10 
        signal_to_noise_raw = power_in_band / (noise_power + epsilon)
        
        # Cap the SNR value before normalization
        capped_snr = min(signal_to_noise_raw, 10.0) # Cap raw SNR

        # Signal Variance
        signal_variance = np.var(filtered_resampled_signal)
        
        # Variance factor
        variance_factor = np.clip(signal_variance * 1000, 0.0, 1.0) 

        # Combine Factors into Quality Score
        w_snr = 0.7
        w_var = 0.3
        
        # Use the capped SNR normalized
        quality_score_unscaled = (w_snr * (capped_snr / 10.0)) + (w_var * variance_factor) 

        # Scale to 0-10
        final_quality_score = np.clip(quality_score_unscaled * 10.0, 0.0, 10.0)

        # Detailed Logging
        print(f"  Quality Calc: P_band={power_in_band:.4e}, P_total={total_power:.4e}, SNR_raw={signal_to_noise_raw:.2f}, SNR_capped={capped_snr:.2f}, Var={signal_variance:.4e}, Var_factor={variance_factor:.3f}, Score={final_quality_score:.2f}")

        return final_quality_score

    except Exception as e:
        print(f"Error calculating signal quality: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def select_best_pos_signal(input_data):
    # Data Preprocessing
    timestamps_np = np.array(input_data["timestamps"])
    region_data_with_timestamps = {}
    for region in ["forehead", "left_cheek", "right_cheek"]:
        if region in input_data and len(input_data[region]) == len(timestamps_np):
            rgb_array = np.array(input_data[region], dtype=float)
            region_data_with_timestamps[region] = {
                'rgb': rgb_array,
                'timestamps': timestamps_np
            }
        else:
            print(f"Skipping region {region} due to missing data or length mismatch.")

    best_filtered_pos_resampled = None
    best_original_rgb = None
    best_original_timestamps = None
    best_uniform_timestamps = None
    best_target_fps = None
    highest_quality = -1.0
    best_region_name = "None"

    for region_name, data in region_data_with_timestamps.items():
        print(f"\nProcessing region: {region_name}")
        rgb_buffer = data.get('rgb')
        timestamps = data.get('timestamps')

        if rgb_buffer is None or timestamps is None:
            print(f"Skipping {region_name}: Missing RGB data or timestamps.")
            continue
        if len(rgb_buffer) != len(timestamps):
            print(f"Skipping {region_name}: Mismatch between RGB ({len(rgb_buffer)}) and timestamp ({len(timestamps)}) count.")
            continue

        # Handle NaN values
        cleaned_rgb, cleaned_timestamps = handle_nan_values(rgb_buffer, timestamps)
        if cleaned_rgb is None:
            print(f"Skipping {region_name}: Insufficient data after NaN removal.")
            continue
        print(f"Region {region_name}: {len(cleaned_timestamps)} samples after NaN handling.")

        try:
            # Apply POS Projection to cleaned data
            pos_signal = apply_pos_projection(cleaned_rgb)
            if pos_signal is None:
                print(f"Skipping {region_name}: POS projection failed.")
                continue
            
            # Timestamp Range Logging
            if len(cleaned_timestamps) > 1:
                time_range = cleaned_timestamps[-1] - cleaned_timestamps[0]
                print(f"DEBUG: Timestamp range for resampling: {time_range:.3f} s (from {cleaned_timestamps[0]:.3f} to {cleaned_timestamps[-1]:.3f})")


            # Resample POS signal to uniform grid
            pos_resampled, uniform_timestamps, target_fps = resample_signal(pos_signal, cleaned_timestamps, target_fps=None)
            if pos_resampled is None or target_fps is None:
                print(f"Skipping {region_name}: Resampling failed.")
                continue
            print(f"Region {region_name}: Resampled to {len(pos_resampled)} samples at {target_fps:.2f} FPS.")

            # Apply Butterworth Filter
            filtered_pos_resampled = apply_butterworth_bandpass(pos_resampled, BAND_MIN_HZ, BAND_MAX_HZ, target_fps)
            if filtered_pos_resampled is None:
                print(f"Skipping {region_name}: Filtering failed.")
                continue

            # Calculate Quality Score
            quality_score = calculate_signal_quality(filtered_pos_resampled, target_fps)
            print(f"Region {region_name}: Quality Score = {quality_score:.2f}")

            # Check if this region is the best
            if quality_score > highest_quality:
                highest_quality = quality_score
                best_filtered_pos_resampled = filtered_pos_resampled
                best_original_rgb = cleaned_rgb
                best_original_timestamps = cleaned_timestamps
                best_uniform_timestamps = uniform_timestamps
                best_target_fps = target_fps
                best_region_name = region_name
                print(f"*** New best region found: {best_region_name} (Quality: {highest_quality:.2f}) ***")

        except Exception as e:
            print(f"Error processing region {region_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if best_filtered_pos_resampled is not None:
        print(f"\nSelected best region: {best_region_name} with quality {highest_quality:.2f}")
        return (best_filtered_pos_resampled, best_original_rgb, best_original_timestamps,
                best_uniform_timestamps, best_target_fps, highest_quality)
    else:
        print("\nNo suitable region found for vital sign extraction.")
        return None, None, None, None, None, -1.0
    
