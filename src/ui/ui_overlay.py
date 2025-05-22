import cv2
import numpy as np

def draw_results_on_frame(vis_frame, last_hr, last_sdnn, last_rmssd, last_br, last_quality, hrv_quality_status):
    y_offset = 30
    overlay_items = []
    if last_hr is not None:
        overlay_items.append(f"HR: {last_hr:.1f} BPM")
    if last_sdnn is not None and not np.isnan(last_sdnn):
        overlay_items.append(f"SDNN: {last_sdnn:.1f} ms")
    if last_rmssd is not None and not np.isnan(last_rmssd):
        overlay_items.append(f"RMSSD: {last_rmssd:.1f} ms")
    if last_br is not None:
        overlay_items.append(f"BR: {last_br:.1f} BPM")
    if last_quality is not None:
        overlay_items.append(f"Quality: {last_quality:.1f}/10.0")
    if hrv_quality_status not in ['ok', 'N/A', None]:
        overlay_items.append(f"HRV Status: {hrv_quality_status}")
    for text in overlay_items:
        cv2.putText(vis_frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y_offset += 20
    return vis_frame
