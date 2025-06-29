import time
from capture_thread import CaptureThread
from ROI.extract_roi import (Extract)
from ROI.visualization import draw
from timeseries.manager import (TimeSeries)
import cProfile
import pstats
from collections import deque
import io
from POS import signal_pipeline, signal_processing
from BR import breathing_pipeline
from STRESS import stress_detection
from PyQt6.QtWidgets import QApplication
from UI.pyqt_ui import AppWindow
import sys


def main_logic(emit_frame, emit_metrics, should_stop):
    roi = Extract()
    series =  TimeSeries()
    capture = CaptureThread("src/vid.avi", debug=True)
    capture.start()
    frame_count = 0
    fps_window = deque(maxlen=30)  # last 30 timestamps
    DETECTION_INTERVAL = 2  # every 2 frames
    last_hr = None
    hr_data = []
    br_data = []
    
    # Signal throttling to prevent UI queue overload
    last_emission_time = 0
    EMISSION_THROTTLE_INTERVAL = 0.5  # 2 emissions to ui per second
    
    rf_model_loaded, scaler_loaded, label_encoder_loaded = stress_detection.load_stress_model_assets()

    try:
        while not should_stop():
            frame, timestamp = capture.get_frame()
            if frame is None:
                if not capture.running.is_set() and not should_stop():
                    break
                time.sleep(0.01)
                continue
            fps_window.append(timestamp)

            if frame_count % DETECTION_INTERVAL == 0:
                roi.process_frame(frame, timestamp)
                vis_frame = draw(frame, roi)
                emit_frame(vis_frame)
            if roi.patches:
                timeseries = series.get(roi.patches, timestamp)
                if timeseries:
                    best_filt, _, best_ts, best_fps, quality, _ = signal_processing.select_best_signal(timeseries)
                    last_hr, last_sdnn, last_rmssd, hrv_quality_status = signal_pipeline.process_hr_from_signal(
                        best_filt, best_ts, best_fps, quality
                    )
                    last_br = breathing_pipeline.process_breathing(
                        best_filt, best_ts, best_fps, quality
                    )
                    
                    if last_hr is not None: hr_data.append((timestamp, last_hr))
                    if last_br is not None: br_data.append((timestamp, last_br))
                    
                    metrics_data = {
                        "timestamp": timestamp,
                        "hr": {"value": last_hr, "unit": "bpm"},
                        "br": {"value": last_br, "unit": "brpm"},
                        "sdnn": {"value": last_sdnn, "unit": "ms"},
                        "rmssd": {"value": last_rmssd, "unit": "ms"},
                        "stress": {"value": None, "unit": ""}
                    }
                    predicted_stress = None
                    if rf_model_loaded and all(v is not None for v in [last_hr, last_sdnn, last_rmssd]):
                        predicted_stress, _ = stress_detection.predict_stress(
                            last_hr, last_sdnn, last_rmssd,
                            rf_model_loaded, scaler_loaded, label_encoder_loaded
                        )
                        if predicted_stress:
                            metrics_data["stress"]["value"] = predicted_stress

                    # Signal throttling to prevent ui queue overload
                    current_time = time.time()
                    if current_time - last_emission_time >= EMISSION_THROTTLE_INTERVAL:
                        emit_metrics(metrics_data)
                        last_emission_time = current_time

            frame_count += 1
            time.sleep(0.01)

    except Exception as e:
        print(f"An error occurred, {e}")

    finally:
        capture.stop()
        print("Processing loop has finished")

        # Save HR data
        if hr_data:
            print(f"Saving {len(hr_data)} HR measurements to hr_data.txt")
            with open("src/hr_data.txt", "w") as f:
                f.write("timestamp,hr\n")
                for ts, hr in hr_data:
                    f.write(f"{ts:.3f},{hr:.2f}\n")
            print("HR data saved successfully!")
            
        # Save BR data
        if br_data:
            print(f"Saving {len(br_data)} Breathing Rate measurements to br_data.txt")
            with open("src/br_data.txt", "w") as f:
                f.write("timestamp,breathing_rate\n")
                for ts, br in br_data:
                    f.write(f"{ts:.3f},{br:.2f}\n")
            print("Breathing Rate data saved successfully!")
        

def profile_block(name, func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(10)  # Show top 10 functions by time
    print(f"\n--- [PROFILE: {name}] ---\n{s.getvalue()}")
    return result


if __name__ == "__main__":    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true', help='Enable profiling for main_logic')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    if args.profile:
        def profiled_logic(emit_frame, emit_metrics, should_stop):
            return profile_block('main_logic', main_logic, emit_frame, emit_metrics, should_stop)
        window = AppWindow(logic_function=profiled_logic)
    else:
        window = AppWindow(logic_function=main_logic)
    window.show()
    sys.exit(app.exec())