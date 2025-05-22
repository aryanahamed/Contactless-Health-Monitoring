import math
from core.config import roi_landmarks
import numpy as np
import cv2

# _______all/Region/dynamic selection calculations/ dont call them_______

# figuring out the yaw angle to determine the face orientation
def yaw_angle(t_matrix):
        matrix = np.array(t_matrix).reshape(4, 4)
        R = matrix[:3, :3]  # 3x3 rotation matrix

        # Extract yaw assuming camera-relative orientation
        yaw = math.atan2(-R[2, 0], R[0, 0])
        return -math.degrees(yaw)  # taking the negative cause our frame is flipped

# dynamically choosing left or right cheek if angle crosses a threshold
def pick_left_or_right(yaw):
    # print(yaw)
    if yaw < -12:
        return "right_cheek"
    elif yaw > 12:
        return "left_cheek"
    else:
        return None # returning none if face is centered/ looking straight


def _raw_cords(landmarks, w, h):
    # Convert all landmarks to NumPy array [num_landmarks, 2]
    return np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)


def _region_cords(all_cords,yaw):
    """
    #converting landmarks to roi coordinates
    returns a dict: {region_name: [(x1, y1), (x2, y2), ...]}
    """

    rmv_region = pick_left_or_right(yaw)
    region_cords = {
        region_name: all_cords[indices].tolist()
        for region_name, indices in roi_landmarks.items()
        if region_name != rmv_region
    }

    return region_cords


def _convex_hull(roi_cords):
    # computing convex hull, returning a dic,basically creating a non_uniform roi border

    hulls = {}

    for region_name, coords in roi_cords.items():
        if len(coords) >= 3:  # needs min 3 dudes, hope it got 3, others its a mess to deal with
            hull = cv2.convexHull(np.array(coords, dtype=np.float32))
            hulls[region_name] = hull
        else:
            hulls[region_name] = np.array(coords, dtype=np.float32)

    return hulls


# using dynamic ema to stabilize landmarks across frames
def ema(curr_cords, prev_cords, base_alpha=.4):
    if prev_cords is None:
        return np.asarray(curr_cords).copy()  # if no prev frame to compare
    else:
        # making sure the data type is correct, should alwyas be but u never know
        curr_cords = np.asarray(curr_cords, dtype=np.float32)
        prev_cords = np.asarray(prev_cords, dtype=np.float32)
        # finding the Euclidean distance between curr and prev of all points
        movement = np.linalg.norm(curr_cords - prev_cords, axis=1)
        # Smooth interpolation for alpha using  fancy tanh, needs more tweaking
        alpha = base_alpha + 0.5 * np.tanh((movement - 1.5) / 2)
        alpha = np.clip(alpha, 0.2, 0.90)
        # brodcasting to  [p,1]
        alpha = alpha[:, np.newaxis]

    return alpha * curr_cords + (1 - alpha) * prev_cords  # usual maths for ema



def extract_patches(frame, hulls):
    fh, fw = frame.shape[:2]
    patches = {}

    for region, hull in hulls.items():
        #some pre checks
        if hull is None or len(hull) < 3:
            patches[region] = None
            continue

        # int conversation, our hull is float32
        hull_int = hull.astype(np.int32)
        x, y, w, h = cv2.boundingRect(hull_int)
        # checking valid bounding box dim
        if w <= 0 or h <= 0:
            patches[region] = None
            continue

        # Clip coordinates to frame boundaries
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, fw), min(y + h, fh)

        # checking if area ok
        if x1 >= x2 or y1 >= y2:
            patches[region] = None
            continue

        #cropping the frame , making mask using that
        cropped_frame = frame[y1:y2, x1:x2]
        ch, cw = cropped_frame.shape[:2] # Cropped h/w
        mask = np.zeros((ch, cw), dtype=np.uint8)

        # 3. Adjust hull relative to the cropped origin (x1, y1), and draw
        relative_hull = hull_int - [x1, y1]
        cv2.fillPoly(mask, [relative_hull], 255)  # type: ignore
        #apply the maskto the cropped frame
        patches[region] = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)

    return patches


