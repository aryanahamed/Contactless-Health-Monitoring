import cv2
import numpy as np


def draw(frame, roi):
    landmarks, regions, valid_rois, thetas, fps_actual, blink, connections = (
        roi.landmarks, roi.region, roi.valid_rois, roi.thetas, roi.fps, roi.attention, roi.landmarker.connections)
    frame = frame.copy()

    if landmarks is not None:
        draw_face_tesselation(frame, landmarks, connections)

    if regions and valid_rois:
        hulls, points = [], []
        for region in valid_rois:
            data = regions.get(region)
            if data is None:
                continue

            if data["hull"] is not None:
                hulls.append(data["hull"].astype(np.int32))
            if data["coordinates"] is not None:
                points.extend(data["coordinates"].astype(np.int32))

        if hulls:
            cv2.polylines(frame, hulls, True, (0, 125, 0), 1)
        if points:
            for pt in points:
                cv2.circle(frame, tuple(pt), 1, (0, 255, 0), -1)

    # Combined text drawing
    # draw_all_text(frame, thetas, fps_actual, blink)
    return frame


def draw_face_tesselation(frame, landmarks, connections):
    landmarks_int = landmarks.astype(np.int32)
    lines = np.array([[landmarks_int[c.start], landmarks_int[c.end]] for c in connections])
    cv2.polylines(frame, lines, False, (200, 200, 200), 1)


# def draw_all_text(frame, thetas, fps_actual, cognitive_data):
#     line_height = 25

#     # Build text array starting with existing metrics
#     texts = [f"FPS: {fps_actual:.1f}",f"Yaw: {int(thetas[0])}",f"Pitch: {int(thetas[1])}",
#         f"Roll: {int(thetas[2])}"
#     ]
#     if cognitive_data:
#         # Attention score
#         attention_score = cognitive_data.get("attention", 0)
#         texts.append(f"Attention: {attention_score:.1f}/100")

#         # Blink metrics
#         blink_data = cognitive_data.get("blinks", {})
#         if blink_data:
#             blink_count = blink_data.get("count", 0)
#             blink_rate = blink_data.get("bpm", 0)
#             texts.append(f"Blinks: {blink_rate:.1f}/min ({blink_count})")

#         # Gaze metrics
#         gaze_data = cognitive_data.get("gaze", ())
#         if len(gaze_data) >= 3:
#             gaze_x, gaze_y, gaze_dev = gaze_data
#             texts.append(f"Gaze: {gaze_x:.2f},{gaze_y:.2f},{gaze_dev:.2f}")

#         # Gaze state
#         gaze_state = cognitive_data.get("gaze_state", "")
#         if gaze_state:

#             texts.append(f"Gaze State: {gaze_state}")

    # Draw all text
    # for i, text in enumerate(texts):
    #     y_pos = 25 + i * line_height

    #     cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


"""
def show(frame,scale):
    if scale != 1:
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('ROI', frame)
"""


