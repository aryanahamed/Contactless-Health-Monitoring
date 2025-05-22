from ROI._landmarker import FaceLandmarkerWrapper
from ROI._segmentation import Segmenter
from ROI._dynamic_roi import _raw_cords,_region_cords,_convex_hull,ema,extract_patches,yaw_angle
import numpy as np
import cv2
import math

# had to use a class here to streamline the whole roi pipeline here
# and not to pass parameters again n again
#also makes main.py much more simpler
# u only need to call process frame, and then access the class values
class ROI:
    def __init__(self):
        #storing all the values that we gonna need for the pipeline each frame calculation
        self._landmarker = FaceLandmarkerWrapper()
        #self.segmentor = Segmenter()
        self.landmarks = None  # current landmarks
        self.mp_image = None  #mp object
        self.roi_cords = {}  #dic to store all roi cords
        self.hulls = {}      # convex hulls dic region wise
        self.mask = None
        self.time = None
        self.h , self.w = None, None  # h,w of frame
        self.prev_all_cords = None    # prev cords for ema
        self.ema_cords = None         #current ema cords
        self.t_matrix = None         #transformation matrix for current frame
        self.patches = None          #actual roi pixels
        self.theta = 0  #face angle x_axis


    def process_frame(self, frame, timestamp):
        self.h, self.w = frame.shape[:2]
        self.time = timestamp
        self.landmarks,self.mp_image,self.t_matrix = self._landmarker.detect(frame, self.time)

        if self.landmarks is None:
            self.roi_cords = {}
            self.hulls = {}
        else:
            self.theta = yaw_angle(self.t_matrix)
            curr_all_cords = _raw_cords(self.landmarks,self.w,self.h)
            #self.ema_cords = ema(curr_all_cords,self.prev_all_cords)
            #self.prev_all_cords = self.ema_cords
            self.roi_cords = _region_cords(curr_all_cords,self.theta)
            self.hulls = _convex_hull(self.roi_cords)
            self.patches = extract_patches(frame,self.hulls)





