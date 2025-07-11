from collections import deque
from ROI.landmarker import FaceLandmarkerWrapper
from ROI.dynamic import get_region,extract_patches, euler_angles
from ROI.stabilization import Stabilization
from ROI.expressions import Expression
from config import regions

class Extract:
    def __init__(self):
        #storing all the values that we are going to need for the pipeline each frame calculation
        self.stabilizer = Stabilization()
        self.landmarker = FaceLandmarkerWrapper()
        self.expressions = Expression()
        self.landmarks = None  # current landmarks
        self.region = {}  #dic to store all roi cords
        self.t_matrix = None         #transformation matrix for current frame
        self.patches = {}         #actual roi pixels
        self.thetas = [0,0,0]  #face euler angles
        self.prev_theta = (0,0,0)
        self.valid_rois = []
        self.fps = 0.0
        self.fps_buffer = deque(maxlen=30)
        self.blendcoff = None
        self.attention = {}
        self.weights ={key: deque(maxlen=5) for key in regions}
        self.count = 0


    def process_frame(self, frame, timestamp):
        h, w = frame.shape[:2]
        self.landmarks,self.t_matrix,self.blendcoff = self.landmarker.detect(frame,h,w,timestamp)
        self.fps = self.get_fps(timestamp, self.fps_buffer)

        if self.landmarks is None:
            self.reset()
        else:
            self.landmarks = self.stabilizer.process(self.landmarks,timestamp)
            self.thetas = euler_angles(self.t_matrix)
            if self.count > 60:
                self.attention = self.expressions.get_attention(self.blendcoff, self.thetas, timestamp)
                self.region = get_region(self.landmarks, self.thetas,self.blendcoff)
                self.patches = extract_patches(frame, self.region, self.weights)
                self.valid_rois = [i for i in self.patches.keys()]
            self.prev_theta = self.thetas
            self.count+=1


    #to get the actual prcessing fps
    @staticmethod
    def get_fps(timestamp, fps_window):
        fps_window.append(timestamp)
        if len(fps_window) >= 2:
            duration = fps_window[-1] - fps_window[0]
            fps = (len(fps_window) - 1) / duration if duration > 0 else 0
        else:
            fps = 0.0

        return fps




    #resettinf all class values
    def reset(self):
        self.landmarks = None
        self.region = {}
        self.t_matrix = None
        self.patches = {}
        self.thetas = (0,0,0)
        self.prev_theta = (0, 0, 0)
        self.valid_rois = []
        self.count=0








