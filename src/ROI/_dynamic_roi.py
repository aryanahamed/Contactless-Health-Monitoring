import math
from config import roi_landmarks
import numpy as np
import cv2
from functools import lru_cache
_prev_theta = [0, 0, 0]
_prev_centers = {}


# _______all/Region/dynamic selection calculations/ don't call them_______

# figuring out the yaw angle to determine the face orientation

def euler_angles(t_matrix):
    matrix = np.array(t_matrix).reshape(4, 4)
    R = matrix[:3, :3]

    # Correct yaw, pitch, roll from rotation matrix
    yaw   = math.atan2(-R[2, 0], R[0, 0])           # ← face sideways
    pitch = math.atan2(R[2, 1], R[2, 2])            # ← face look up or down
    roll  = math.atan2(R[1, 0], R[0, 0])            # ← face tilt to shoulder

    return round(-math.degrees(yaw)), round(-math.degrees(pitch)), round(-math.degrees(roll))



def _region_cords(all_cords,e_angle):
    """
    #converting landmarks to roi coordinates
    returns a dict: {region_name: [(x1, y1), (x2, y2), ...]}
    """
    rmv = remove_regions(e_angle)
    return {
        r: all_cords[idx].tolist()
        for r, idx in roi_landmarks.items()
        if r not in rmv
    }

def _bounded_hull(roi_cords, fw, fh, prev_boxes, scale=0.9, thetas=(0, 0, 0)):
    global _prev_theta, _prev_centers
    results = {}

    for region, cords in roi_cords.items():
        if len(cords) < 3:
            results[region] = None
            continue

        # Convex hull
        hull = cv2.convexHull(np.array(cords, dtype=np.float32))

        # Bounding box around the convex hull (ensures tight containment)
        x, y, w, h = cv2.boundingRect(hull)
        cx = x + w // 2
        cy = y + h // 2
        w = int(w * scale)
        h = int(h * scale)

        # Threshold check
        if not _should_update_roi(region, (cx, cy), thetas):
            last = prev_boxes.get(region)
            if last and "bbox" in last and "hull" in last:
                results[region] = {
                    "bbox": last["bbox"],
                    "hull": last["hull"],
                    "center": (last["cx"], last["cy"]),
                }
            else:
                results[region] = None
            continue

        # Smooth
        cx, cy, w, h = tuple(map(lambda p: int(round(p)),
                                 smooth_box_ema(region, cx, cy, w, h, prev_boxes)))
        x1, y1, x2, y2 = clip_bbox(cx, cy, w, h, fw, fh)

        if x2 <= x1 or y2 <= y1:
            results[region] = None
            continue

        bbox = (x1, y1, x2, y2)

        results[region] = {
            "bbox": bbox,
            "hull": hull,
            "center": (cx, cy),
        }

        # Update state
        prev_boxes[region].update({
            "cx": cx, "cy": cy, "w": w, "h": h,
            "bbox": bbox, "hull": hull
        })
        _prev_centers[region] = (cx, cy)

    if any(results[r] is not None and results[r]["bbox"] != prev_boxes.get(r, {}).get("bbox") for r in roi_cords):
        _prev_theta = thetas
    return results


# using dynamic ema to stabilize landmarks across frames
def ema(curr_cords,prev_cords, base_alpha=.1,m = 3,b=2):
    if prev_cords is None:
        return curr_cords # if no prev to compare
    else:
        # finding the Euclidean distance between curr and prev of all points/point
        movement = np.linalg.norm(curr_cords - prev_cords, axis=1)
        # Smooth interpolation for alpha using  fancy tan-h, needs more tweaking
        alpha = base_alpha + 0.7 * np.tanh((movement - m) / b)
        alpha = np.clip(alpha, 0.05, 0.90)
        # broadcasting to  [p,1]
        alpha = alpha[:, np.newaxis]

    return alpha * curr_cords + (1 - alpha) * prev_cords  # usual maths for ema


def extract_patches(frame, b_hulls):
    """
    getting the roi patches from _bounded_hull().
    returns-dict: {region_name: patch (H, W, 3) or None}
    """
    patches = {}
    for r, info in b_hulls.items():
        if info is None:
            patches[r] = None
            continue
        x1, y1, x2, y2 = info["bbox"]
        patch = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
        patches[r] = patch if is_skin(patch) else None

    return patches


def smooth_box_ema(region, cx, cy, w, h, prev_boxes, smooth_size=True):
    box = prev_boxes.setdefault(region, {"cx": cx, "cy": cy, "w": w, "h": h})

    cx = update(cx, box["cx"])
    cy = update(cy, box["cy"])
    if smooth_size:
        w = update(w, box["w"])
        h = update(h, box["h"])

    box.update({"cx": cx, "cy": cy, "w": w, "h": h})
    return cx, cy, w, h


@lru_cache(maxsize=128)
def remove_regions(e_angle):
    yaw, pitch, roll = tuple(e_angle)
    to_remove = set()

    # Yaw-based filtering
    if yaw <= -25: to_remove.update(["right_cheek", "forehead"])
    elif yaw >= 25: to_remove.update(["left_cheek", "forehead"])
    elif yaw <= -15: to_remove.add("right_cheek")
    elif yaw >= 15: to_remove.add("left_cheek")

    # Pitch-based (looking up/down too much)
    if abs(pitch) > 25 or abs(roll) > 15:
        to_remove.update(["forehead", "right_cheek", "left_cheek"])

    return tuple(to_remove)

#clip the bbox
def clip_bbox(cx, cy, w, h, fw, fh):
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    return x1, y1, min(fw, x1 + w), min(fh, y1 + h)

def update(current, prev):
    curr = np.array([[current]], dtype=np.float32)
    prev = np.array([[prev]], dtype=np.float32)
    return ema(curr, prev,base_alpha=.1,m=3,b=2).item()


def is_skin(roi, threshold=0.5):
    if roi is None or roi.size == 0:
        return False

    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    # Skin color bounds (based on empirical research)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    # Skin mask and skin pixel ratio
    mask = cv2.inRange(ycrcb, lower, upper)

    # Return True if skin dominates the patch
    return np.count_nonzero(mask) / mask.size >= threshold



def _should_update_roi(region, center, current_theta, yaw_thresh=3, pitch_thresh=3, roll_thresh=4, center_thresh=3):
    if region not in _prev_centers:
        return True

    cx, cy = center
    pcx, pcy = _prev_centers[region]

    center_deltas = (abs(cx - pcx), abs(cy - pcy))
    angle_deltas = (
        abs(current_theta[0] - _prev_theta[0]),
        abs(current_theta[1] - _prev_theta[1]),
        abs(current_theta[2] - _prev_theta[2])
    )

    return (center_deltas[0] > center_thresh or center_deltas[1] > center_thresh or
            angle_deltas[0] > yaw_thresh or angle_deltas[1] > pitch_thresh or angle_deltas[2] > roll_thresh)