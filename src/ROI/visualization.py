import numpy as np
import cv2

def draw_points(frame, roi_dict, theta):
    output = np.copy(frame)
    point_color = (0, 255, 0)  # max white

    label_color = (255, 255, 255)
    label_alpha = 0.35
    label_font_scale = 0.4
    label_thickness = 1

    label_overlay = output.copy()

    for region, raw_points in roi_dict.items():
        if not raw_points:
            continue

        pts_array = np.array(raw_points, dtype=np.int32)

        # Draw points
        for x, y in pts_array:
            cv2.circle(output, (x, y), radius=2, color=point_color, thickness=-1)

        # Get label
        if region == "left_cheek":
            label = "Left"
        elif region == "right_cheek":
            label = "Right"
        else:
            label = region.replace("_", " ").title()

        # Label position
        cx, cy = np.mean(pts_array, axis=0).astype(int)
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
        label_x = cx - text_width // 2
        label_y = cy - 10

        cv2.putText(label_overlay, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_color, label_thickness, cv2.LINE_AA)

    # Blend translucent labels
    output = cv2.addWeighted(label_overlay, label_alpha, output, 1 - label_alpha, 0)

    # Draw yaw arrow
    if theta is not None:
        output = draw_yaw_arrow_on_frame(output, roi_dict, theta)

    return output


def draw_hulls(frame,hull):
    if not hull:
        return frame
    frame = frame.copy()
    for region, points in hull.items():
        points_int = points.astype(np.int32)
        cv2.polylines(frame, [points_int], isClosed=True, color=(0, 255, 0), thickness=1)
        #cv2.fillPoly(frame, [points_int], color=(0, 255, 0))
    return frame




def apply_mask_to_frame(frame_shape, mask):
    result = np.zeros(frame_shape, dtype=np.uint8)

    if mask is None:
        return result

    if mask.ndim == 3:
        mask = np.squeeze(mask)
    result[mask == 1] = [0, 0, 255]       # Hair - red
    result[mask == 2] = [0, 0, 255]       # Body skin - red
    result[mask == 3] = [255, 255, 255]   # Face - white
    # background (0) remains black
    return result



def draw_yaw_arrow_on_frame(frame, roi_dict, theta_deg):
    if "forehead" not in roi_dict or len(roi_dict["forehead"]) == 0:
        return frame

    # Forehead center point
    forehead_pts = np.array(roi_dict["forehead"])
    cx, cy = np.mean(forehead_pts, axis=0).astype(int)

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

    color = 255, 255, 255

    # Draw arrow and yaw text
    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
    cv2.putText(frame, f"{round(theta_deg,2)}", (cx + 10, cy - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    return frame





