import numpy as np
import cv2
from matplotlib import pyplot as plt




def draw_hulls(frame, hulls,cords,valid_rois,theta,fps_actual):
    frame = frame.copy()
    if not hulls or not valid_rois:
        return frame


    for region, data in hulls.items():
        if data is None or region not in valid_rois:
            continue

        [cv2.circle(frame, tuple(map(int, pt)), 1, (0, 255, 0), -1) for pt in cords[region]]

        # Draw bounding box (blue rectangle)
        bbox = data.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)

    #draw yaw arrow
    draw_yaw_arrow_on_frame(frame,theta)
    fps_text = f"FPS: {fps_actual:.2f}"
    cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def draw_yaw_arrow_on_frame(frame, theta_deg):
    """
    Draws a horizontal yaw arrow near the top center of the frame, with direction and length based on yaw angle.
    """
    h, w = frame.shape[:2]

    # Define origin point near top-center
    cx = w // 2
    cy = 40  # distance from top (adjust as needed)

    # Arrow length scaling
    abs_theta = abs(theta_deg)
    min_len = 10
    max_len = 60
    scale = min(max(abs_theta / 30, 0), 1)  # Normalize between 0 and 1 for up to 30°
    arrow_len = int(min_len + scale * (max_len - min_len))  # Grow with angle

    # Arrow direction (follows face turn)
    dx = arrow_len if theta_deg > 0 else -arrow_len
    dy = 0

    end_x = cx + dx
    end_y = cy + dy

    color = (255, 255, 255)

    # Draw arrow and yaw text
    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
    cv2.putText(frame, f"{round(theta_deg, 2)}", (cx - 20, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame




class LongTermCollector:
    def __init__(self):
        self.raw_series = {}
        self.interp_series = {}
        self.last_timestamp = None  # To detect duplicates

    def append(self, uniform_series, series_np):
        # Get last index (most recent frame in the sliding buffer)
        idx = -1
        ts_new = series_np["timestamps"][idx]
        if ts_new == self.last_timestamp:
            return  # skip duplicate
        self.last_timestamp = ts_new

        for region in series_np:
            if region == "timestamps":
                self.raw_series.setdefault("timestamps", []).append(series_np["timestamps"][idx])
                self.interp_series.setdefault("timestamps", []).append(uniform_series["timestamps"][idx])
            else:
                self.raw_series.setdefault(region, []).append(series_np[region][idx])
                self.interp_series.setdefault(region, []).append(uniform_series[region][idx])

    def get_combined_series(self):
        return (
            {k: np.array(v) for k, v in self.raw_series.items()},
            {k: np.array(v) for k, v in self.interp_series.items()}
        )


def plot_interpolation_comparison(uniform_series, series_np, region='forehead'):

    if region not in uniform_series or region not in series_np:
        print(f"Region '{region}' not found.")
        return

    # Extract timestamps
    t_orig = np.array(series_np["timestamps"])
    t_interp = np.array(uniform_series["timestamps"])

    # Extract RGB values
    v_orig = np.array(series_np[region])  # shape: (N, 3)
    v_interp = np.array(uniform_series[region])  # shape: (N, 3)

    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(12, 5))

    for ch in range(3):
        # Interpolated curve
        plt.plot(t_interp, v_interp[:, ch], '-', label=f'{colors[ch]} interp', linewidth=2)
        # Original raw values (possibly with NaNs)
        plt.plot(t_orig, v_orig[:, ch], 'o', alpha=0.4, label=f'{colors[ch]} raw')

    plt.title(f"RGB Interpolation vs Raw – Region: {region}")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def w_i_plot(u,series,region):
    t_raw = np.array(series["timestamps"])
    t_interp = np.array(u["timestamps"])
    raw = np.array(series[region])  # (N, 3)
    interp = np.array(u[region])  # (M, 3)

    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(10, 5))
    for ch in range(3):
        plt.plot(t_interp, interp[:, ch], '-', label=f'{colors[ch]} interp', linewidth=2)
        plt.plot(t_raw, raw[:, ch], 'o', alpha=0.5, label=f'{colors[ch]} raw')
    plt.title(f"[DEBUG] One Window – PCHIP vs Raw: {region}")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()





