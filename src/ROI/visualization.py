import cv2
import numpy as np


def draw(frame, roi):
    landmarks, regions, valid_rois, thetas, fps_actual, blink, connections = (
        roi.landmarks, roi.region, roi.valid_rois, roi.thetas, roi.fps, roi.blink, roi.landmarker.connections)
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
    draw_all_text(frame, thetas, fps_actual, blink)
    return frame


def draw_face_tesselation(frame, landmarks, connections):
    landmarks_int = landmarks.astype(np.int32)
    lines = np.array([[landmarks_int[c.start], landmarks_int[c.end]] for c in connections])
    cv2.polylines(frame, lines, False, (128, 128, 128), 1)


def draw_all_text(frame, thetas, fps_actual, cognitive_data):
    line_height = 25

    # Build text array starting with existing metrics
    texts = [
        f"FPS: {fps_actual:.1f}",
        f"Yaw: {int(thetas[0])}",
        f"Pitch: {int(thetas[1])}",
        f"Roll: {int(thetas[2])}"
    ]

    # Add cognitive metrics if available
    if cognitive_data:
        blink_data = cognitive_data.get("blink", {})
        cognitive_state = cognitive_data.get("cognitive", {})

        # Blink metrics
        blink_count = blink_data.get("blink_count", 0)
        blink_rate = blink_data.get("blink_pm", 0)
        texts.append(f"Blinks: {blink_count} ({blink_rate:.1f}/min)")

        # Stress and attention
        stress_level = cognitive_state.get("stress_level", "-")
        stress_score = cognitive_state.get("stress_score", 0)
        texts.append(f"Stress: {stress_level} ({stress_score:.2f})")

        attention_level = cognitive_state.get("attention_level", "-")
        attention_score = cognitive_state.get("attention_score", 0)
        texts.append(f"Attention: {attention_level} ({attention_score:.2f})")

        # Z-scores from details
        details = cognitive_state.get("details", {})
        gaze_z = details.get("gaze_z", 0)
        eye_opening_z = details.get("eye_opening_z", 0)
        eye_strain_z = details.get("eye_strain_z", 0)
        texts.append(f"Gaze: {gaze_z:.2f}")
        texts.append(f"Eye Opening: {eye_opening_z:.2f}")
        texts.append(f"Eye Strain: {eye_strain_z:.2f}")

        # Status (with progress only during baseline)
        status = cognitive_state.get("status", "-")
        progress = cognitive_state.get("progress", 0)
        if status == "establishing_baseline":
            texts.append(f"Status: ({progress:.0f}%)")
        else:
            texts.append(f"Status: {status}")

    # Draw all text
    for i, text in enumerate(texts):
        y_pos = 25 + i * line_height
        cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)



def show(frame,scale):
    if scale != 1:
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('ROI', frame)



