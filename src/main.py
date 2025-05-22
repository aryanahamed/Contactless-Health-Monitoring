from capture_thread import CaptureThread, wait_for_startup
from ROI.extract_roi import ROI
from ROI.visualization import draw_points
from timeseries.manager import Manager
import cv2
import matplotlib.pyplot as plt
from core.signal_pipeline import process_signals
from ui.ui_overlay import draw_results_on_frame
from core.config import GROUND_TRUTH
import numpy as np
from debug.ground_truth_analysis import analyze_ground_truth_vs_calculated, load_ground_truth_hr


def main():
    roi_class = ROI()
    series_generator = Manager()
    capture = CaptureThread()
    capture.start()
    wait_for_startup(capture, delay_sec=0)
    enable_signal_debug = True
    signal_figure, signal_ax, br_ax, rgb_diag_fig, rgb_diag_axs = None, None, None, None, None
    if enable_signal_debug:
        from debug.debug_plots import setup_debug_figures
        signal_figure, signal_ax, br_ax, rgb_diag_fig, rgb_diag_axs = setup_debug_figures()
    vis_frame = None
    ground_truth_hr = load_ground_truth_hr(GROUND_TRUTH)
    calculated_hr = []
    frame_idx = 0
    while True:
        frame, timestamp = capture.get_frame()
        if frame is None:
            break
        roi_class.process_frame(frame, timestamp)
        vis_frame = draw_points(frame, roi_class.roi_cords, roi_class.theta) if roi_class.roi_cords is not None else frame.copy()
        series = series_generator.get_series(roi_class.patches, timestamp) if roi_class.patches else None
        last_hr = last_sdnn = last_rmssd = last_br = last_quality = None
        hrv_quality_status = 'N/A'
        if series is not None:
            last_hr, last_sdnn, last_rmssd, last_br, last_quality, hrv_quality_status = process_signals(
                series, enable_signal_debug, br_ax, signal_ax=signal_ax, rgb_diag_axs=rgb_diag_axs)
        calculated_hr.append(last_hr if last_hr is not None else np.nan)
        frame_idx += 1
        vis_frame = draw_results_on_frame(vis_frame, last_hr, last_sdnn, last_rmssd, last_br, last_quality, hrv_quality_status)
        cv2.imshow('ROI', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if enable_signal_debug and signal_figure is not None:
        plt.close(signal_figure)
    if enable_signal_debug and rgb_diag_fig is not None:
        plt.close(rgb_diag_fig)
    capture.stop()
    cv2.destroyAllWindows()
    analyze_ground_truth_vs_calculated(ground_truth_hr, calculated_hr)


if __name__ == "__main__":
    main()