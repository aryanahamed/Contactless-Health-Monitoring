# main.py
import numpy as np
import cv2
import time
from capture_thread import CaptureThread,wait_for_startup
from ROI.extract_roi import (Extract)
from ROI.visualization import draw_hulls,plot_interpolation_comparison,LongTermCollector
from timeseries.manager import (Manager)
import cProfile
import pstats
from collections import deque
import io
from POS import signal_pipeline
from BR import breathing_pipeline
from STRESS import stress_detection
import heartpy as hp
import pandas as pd
import matplotlib.pyplot as plt

fps_window = deque(maxlen=30)  # last 30 timestamps
DETECTION_INTERVAL = 2  # every 2 frames

def main():
    roi_class = Extract()
    series_generator =  Manager()
    capture = CaptureThread()
    capture.start()
    #collector = LongTermCollector()
    # wait_for_startup(capture, delay_sec=3)
    frame_count = 0
    last_hr = None
    hr_data = []
    br_data = []
    sdnn_data = []
    rmssd_data = []
    
    rf_model_loaded, scaler_loaded, label_encoder_loaded = stress_detection.load_stress_model_assets()

    try:
        while True:
            frame, timestamp = capture.get_frame()
            if frame is None:
                if not capture.running.is_set():
                    break
                time.sleep(0.01)
                continue
            fps_window.append(timestamp)

            if len(fps_window) >= 2:
                duration = fps_window[-1] - fps_window[0]
                fps_actual = (len(fps_window) - 1) / duration if duration > 0 else 0
            else:
                fps_actual = 0.0
            if frame_count % DETECTION_INTERVAL == 0:
                roi_class.process_frame(frame, timestamp)
            vis_frame = draw_hulls(frame, roi_class.hulls,roi_class.region_cords,
                                   roi_class.valid_rois, roi_class.thetas[0], fps_actual)
            cv2.imshow('ROI', vis_frame)
            if roi_class.patches:
                series = series_generator.get_series(roi_class.patches, timestamp)
                if series:
                    last_hr, last_sdnn, last_rmssd, quality, hrv_quality_status = signal_pipeline.process_hr(series)
                    last_br = breathing_pipeline.process_breathing(series)

                    # # Heartpy implementation
                    # forehead = series["forehead"][:, 1]
                    # filtered_signal = hp.filter_signal(forehead, cutoff=[0.67, 3.0], sample_rate=30, order=3, filtertype='bandpass')
                    # working_data, measures = hp.process(filtered_signal, 30.0)
                    # print("Heart rate(HeartPy):", measures['bpm'])
                    # hr_data.append((timestamp, measures['bpm']))


                    #collector.append(u, series)
                    #raw_all, uniform_all = collector.get_combined_series()
                    #if len(raw_all.get("forehead", [])) >= 1000:
                        #plot_interpolation_comparison(uniform_all, raw_all, region="forehead")
                        #break
                    
                    if last_hr is not None:
                        print(f"HR: {round(last_hr)}")
                        hr_data.append((timestamp, last_hr))
                    if last_br is not None:
                        print(f"Breathing Rate: {round(last_br)}")
                        br_data.append((timestamp, last_br))
                    if last_sdnn is not None:
                        print(f"SDNN: {round(last_sdnn, 2)} ms")
                        sdnn_data.append((timestamp, last_sdnn))
                    if last_rmssd is not None:
                        print(f"RMSSD: {round(last_rmssd, 2)} ms")
                        rmssd_data.append((timestamp, last_rmssd))
                        print(f"HRV Quality: {hrv_quality_status}")
                    if quality is not None:
                        print(f"Quality Score: {quality:.2f}")
                
                if rf_model_loaded and scaler_loaded and label_encoder_loaded:
                    if last_hr is not None and last_sdnn is not None and last_rmssd is not None:
                        predicted_stress, _ = stress_detection.predict_stress(
                            last_hr, last_sdnn, last_rmssd,
                            rf_model_loaded, scaler_loaded, label_encoder_loaded
                        )
                        if predicted_stress:
                            print(f"Stress Intensity: {predicted_stress}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1

    except KeyboardInterrupt:
        pass

    finally:
        capture.stop()
        cv2.destroyAllWindows()

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
    # main()
    profile_block("Main Function", main)