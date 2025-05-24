from ROI._landmarker import FaceLandmarkerWrapper
from ROI._dynamic_roi import _region_cords,_bounded_hull,ema,extract_patches,euler_angles
import numpy as np

# had to use a class here to streamline the whole roi pipeline here
# and not to pass parameters again n again
#also makes main.py much more simpler
# u only need to call process frame, and then access the class values
class Extract:
    def __init__(self):
        #storing all the values that we are going to need for the pipeline each frame calculation
        self._landmarker = FaceLandmarkerWrapper()
        self.landmarks = None  # current landmarks
        self.region_cords = {}  #dic to store all roi cords
        self.hulls = {}      # convex hulls dic region wise
        self.prev_all_cords = None    # prev cords for ema
        self.ema_cords = None         #current ema cords
        self.t_matrix = None         #transformation matrix for current frame
        self.patches = {}         #actual roi pixels
        self.thetas = (0,0,0)  #face euler angles
        self.prev_boxes = {}
        self.valid_rois = []


    def process_frame(self, frame, timestamp):
        self.landmarks,self.t_matrix = self._landmarker.detect(frame, timestamp)

        if self.landmarks is None:
            self._reset()
        else:
            h, w = frame.shape[:2]
            self.thetas = euler_angles(self.t_matrix)
            raw_cords = np.array([[lm.x * w, lm.y * h] for lm in self.landmarks], dtype=np.float32)
            self.ema_cords = ema(raw_cords,self.prev_all_cords)
            self.region_cords = _region_cords(self.ema_cords, self.thetas)
            self.hulls = _bounded_hull(self.region_cords, w, h,self.prev_boxes,thetas=self.thetas)
            self.patches = extract_patches(frame,self.hulls)
            self.valid_rois = [i for i, j in self.patches.items() if j is not None]
            self.prev_all_cords = self.ema_cords
            #if self.valid_rois:




    #resettinf all class values
    def _reset(self):
        self.landmarks = None
        self.region_cords = {}
        self.hulls = {}
        self.prev_all_cords = None
        self.ema_cords = None
        self.t_matrix = None
        self.patches = {}
        self.thetas = (0,0,0)
        self.prev_boxes = {}
        self.valid_rois = []






