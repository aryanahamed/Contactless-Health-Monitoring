import numpy as np
from config import anchor_indices


#one euro filter for stabilizing the landmarks, could directly import a extenal libray but oh well, its cool
class Stabilization:
    ##standard one euro implementation with some tweaks for our specific case
    def __init__(self):
        self.anchor_indices = anchor_indices
        self.min = 0.1
        self.beta = 0.2
        self.dcut = 0.5

        self.prevlands = None
        self.prev_anchor_velocity = 0.0
        self.prev_time = None
        self.landmark_prev_values = None


    def calculate_anchor_velocity(self, current_landmarks, dt):
        if self.prevlands is None:
            self.prevlands = current_landmarks
            return 0.0

        if dt <= 0:
            return self.prev_anchor_velocity

        # Extract anchor
        current_anchors = current_landmarks[self.anchor_indices]
        prev_anchors = self.prevlands[self.anchor_indices]
        #displacement anchor
        displacements = np.linalg.norm(current_anchors - prev_anchors, axis=1)
        median_displacement = np.median(displacements)
        #velocity (pixels/second)
        velocity = median_displacement / dt
        #estimate
        alpha_v = 1.0 - np.exp(-2.0 * np.pi * self.dcut * dt)
        smoothed_velocity = alpha_v * velocity + (1.0 - alpha_v) * self.prev_anchor_velocity
        # Update
        self.prevlands = current_landmarks
        self.prev_anchor_velocity = smoothed_velocity

        return smoothed_velocity

    def process(self, current_landmarks, timestamp):
        if self.prev_time is None:
            self.prev_time = timestamp
            self.landmark_prev_values = current_landmarks
            self.prevlands = current_landmarks
            return current_landmarks

        dt = timestamp - self.prev_time
        anchor_velocity = self.calculate_anchor_velocity(current_landmarks, dt)
        smoothed_landmarks = np.zeros_like(current_landmarks)
        cutoff = self.min + self.beta * anchor_velocity
        alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff * dt)
        alpha = np.clip(alpha, 0.0, 1.0)
        smoothed_landmarks = alpha * current_landmarks + (1.0 - alpha) * self.landmark_prev_values

        self.landmark_prev_values = smoothed_landmarks
        self.prev_time = timestamp

        return smoothed_landmarks



#tanh ema logic for the entire pipeline
def ema(curr, prev, base_alpha=.1, m=3, b=2):
    if prev is None:return curr

    if np.isscalar(curr):
        movement = abs(curr - prev)
    else:
        movement = np.linalg.norm(curr - prev, axis=1)
    alpha = base_alpha + 0.7 * np.tanh((movement - m) / b)
    alpha = np.clip(alpha, 0.05, 0.90)
    if not np.isscalar(curr):
        alpha = alpha[:, np.newaxis]
    return alpha * curr + (1 - alpha) * prev