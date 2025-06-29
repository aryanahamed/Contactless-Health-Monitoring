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

    return frame


def draw_face_tesselation(frame, landmarks, connections):
    landmarks_int = landmarks.astype(np.int32)
    lines = np.array([[landmarks_int[c.start], landmarks_int[c.end]] for c in connections])
    cv2.polylines(frame, lines, False, (128, 128, 128), 1)




def show(frame,scale):
    if scale != 1:
        h, w = frame.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('ROI', frame)