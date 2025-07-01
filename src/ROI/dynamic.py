import math
from config import roi_indices
import numpy as np
import cv2
from ROI.stabilization import ema
from ROI.expressions import Expression

# _______all/Region/dynamic selection calculations/ don't call them_______

# figuring out the yaw angle to determine the face orientation
def euler_angles(t_matrix):
    if t_matrix is None or t_matrix.size <2:
        return 0,0,0

    yaw   = math.atan2(t_matrix[2, 0], t_matrix[0, 0])           #face sideways
    pitch = math.atan2(t_matrix[2, 1], t_matrix[2, 2])            #face look up or down
    roll  = math.atan2(t_matrix[1, 0], t_matrix[0, 0])            #face tilt to shoulder

    return math.degrees(yaw),-math.degrees(pitch), math.degrees(roll)


def get_region(all_cords, e_angle, blendshape):
    rmv = set(remove_euler(e_angle) + Expression.get_excluded_rois(blendshape))

    results = {}
    for r, landmark_indices in roi_indices.items():
        if r in rmv:
            continue

        cords = all_cords[landmark_indices]
        hull = cv2.convexHull(cords.astype(np.float32))
        bbox = cv2.boundingRect(hull)
        results[r] = {'coordinates': cords,'hull': hull,'bbox': bbox}

    return results


def extract_patches(frame, roi_dict, weights):
    results = {}
    frame_height, frame_width = frame.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for region, data in roi_dict.items():
        if data is None:
            continue

        hull = data['hull']
        x, y, w, h = data['bbox']

        # bounds check
        if (w <= 0 or h <= 0 or
                x < 0 or y < 0 or
                x >= frame_width or y >= frame_height or
                x + w > frame_width or y + h > frame_height):
            continue

        # frame cropping
        frame_crop = frame[y:y + h, x:x + w]
        if frame_crop.size == 0:
            continue

        #local hull mask
        local_hull = (hull - [x, y]).astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [local_hull], (255,))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        area_px = cv2.countNonZero(mask)

        if area_px == 0 or is_occluded(frame_crop):
            continue

        # calc weight
        if len(weights[region]) == 0:
            weight = np.sqrt(area_px)
            weights[region].append(weight)
        else:
            prev_weight = np.mean(weights[region])
            weight = ema(np.sqrt(area_px), prev_weight, base_alpha=.1, m=100)
            weights[region].append(weight)

        # Extract hull-based RGB mean
        mean_bgr = cv2.mean(frame_crop, mask=mask)[:3]
        mean_rgb = mean_bgr[::-1]

        results[region] = {
            "rgb": mean_rgb,
            "weight": weight,
        }

    return results


def remove_euler(e_angle):
    yaw, pitch, roll = tuple(e_angle)
    to_remove = set()
    # using yaw to remove regions
    if yaw <= -20: to_remove.update(["right_cheek", "forehead"])
    elif yaw >= 20: to_remove.update(["left_cheek", "forehead"])
    elif yaw <= -10: to_remove.add("right_cheek")
    elif yaw >= 10: to_remove.add("left_cheek")
    #pitch/roll-based removal
    if abs(pitch) > 15 or abs(roll) > 20:
        to_remove.update(["forehead", "right_cheek", "left_cheek"])

    return list(to_remove)


def is_occluded(frame_crop):
    ycrcb = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2YCrCb)
    low = np.array([0, 125, 70], dtype=np.uint8) ##these are taken from papers
    upper = np.array([255, 180, 135], dtype=np.uint8)#using slightly wider range
    skin_maMsk = cv2.inRange(ycrcb, low, upper)
    skin_pixels = cv2.countNonZero(skin_maMsk)
    total_pixels = frame_crop.shape[0] * frame_crop.shape[1]
    return skin_pixels / total_pixels < 0.6
