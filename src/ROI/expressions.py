from collections import deque
import numpy as np

class Expression:
    def __init__(self, thresh=0.5, win=61):
        self.b_thresh = thresh
        self.win = win  # window in seconds
        self.ts = deque()  # timestamps
        self.blink_vals = deque()
        self.b_times = deque()  # only timestamps of actual blinks
        self.last_b_val = 0
        # seems to work for adapting the baseline
        self.bpm_hist = deque(maxlen=120)
        # stoing if in a blink or not
        self.in_a_blink = False
        self.blink_start_time = 0
        # gotta store the duration
        self.durs = deque()
        # track if the eye is closed
        self.is_closed_hist = deque()
        self.is_closed_ts = deque()
        self.smooth_att = 75.0  # start at a neutral 75, not 0
        self.is_off = False
        self.off_target_t = 0
        self.g_hist = deque(maxlen=5) #for fixation
        self.g_state = 'START'
        self.low_v_count = 0
        # for the continuous baseline stuff
        self.good_gaze_devs = deque(maxlen=300)

    def blink(self, blendshapes, timestamp):
        # this is kinda janky but it works. blink is when both vals dip
        b_val = max(blendshapes[8], blendshapes[9])
        self.ts.append(timestamp)  # pop the old stugg
        self.blink_vals.append(b_val)
        while self.ts and timestamp - self.ts[0] > self.win:
            self.ts.popleft()
            self.blink_vals.popleft()
        while self.b_times and timestamp - self.b_times[0] > self.win:
            self.b_times.popleft()
            self.durs.popleft()

        # perclos
        while self.is_closed_ts and timestamp - self.is_closed_ts[0] > self.win:
            self.is_closed_ts.popleft()
            self.is_closed_hist.popleft()

        # if we aare currently in ablink or not
        currently_closed = b_val > self.b_thresh
        # check for state changes.
        if currently_closed and not self.in_a_blink:
            self.in_a_blink = True
            self.blink_start_time = timestamp
        elif not currently_closed and self.in_a_blink:
            self.in_a_blink = False
            self.b_times.append(timestamp)

            # get the blink duration
            duration = timestamp - self.blink_start_time
            self.durs.append(duration)
        blink_now = self.last_b_val < self.b_thresh <= b_val  # flipped this to be a rising edge for consistency
        self.last_b_val = b_val
        # gotta update the PERCLOS history every single frame
        self.is_closed_hist.append(1 if currently_closed else 0)
        self.is_closed_ts.append(timestamp)
        win_duration = (self.ts[-1] - self.ts[0]) if len(self.ts) > 1 else 1.0
        # calculate the bpm
        bpm = (len(self.b_times) / win_duration) * 60 if win_duration > 0 else 0.0
        self.bpm_hist.append(bpm)
        if len(self.b_times) > 1:
            ibi = np.diff(np.array(self.b_times))
            mean_ibi = float(np.mean(ibi))
            cv = float(np.std(ibi)) / mean_ibi if mean_ibi > 0 else 0.0
        else:
            cv = 0.0  # no variability if there's no blinks
        # avg duration of blinks in the window
        avg_dur = float(np.mean(self.durs)) if self.durs else 0.0
        # print(avg_dur)
        perclos = float(np.mean(self.is_closed_hist)) if self.is_closed_hist else 0.0
        # print(ibi)

        return {
            'bpm': bpm, 'count': len(self.b_times), 'cv': cv, 'is_blinking': currently_closed,
            'avg_duration': avg_dur,
            'perclos': perclos
        }

    ##gotta build atteniton here and link it with blink, might create a another funciton to get both
    # stick this in the Expression class
    def pose(self, blend):
        y, p = blend[0], blend[1]
        # idk these numbers feel about right. can change em later
        y_ok, ymax, p_ok, p_max = 20, 35, 25, 40
        y_pen, p_pen = 0, 0  # penalty
        if abs(y) > y_ok:
            how_far = abs(y) - y_ok
            total_range = ymax - y_ok
            y_pen = how_far / total_range

        if abs(p) > p_ok:
            how_far = abs(p) - p_ok
            total_range = p_max - p_ok
            p_pen = how_far / total_range
        # just take the biggest penalty of the twol,
        pen = max(p_pen, y_pen)
        if pen > 1.0: pen = 1.0
        score = (1.0 - pen) * 100
        return score

    def gaze(self, b_shapes):
        # these are the directions,taking from congig
        look_R = max(b_shapes[15], b_shapes[12])
        look_L = max(b_shapes[14], b_shapes[13])
        gaze_x = look_R - look_L
        look_U = max(b_shapes[16], b_shapes[17])
        look_D = max(b_shapes[10], b_shapes[11])
        gaze_y = look_U - look_D
        dev = (gaze_x ** 2 + gaze_y ** 2) ** 0.5  ## now the pyth thing to get total offcenter amount
        return gaze_x, gaze_y, dev

    # combining it all here, prolly just gonna call this from extract cclass
    def get_attention(self, b_shapes, angs, ts):
        blinks = self.blink(b_shapes, ts)
        h_score = self.pose(angs)
        g_vec = self.gaze(b_shapes)
        g_x, g_y, g_dev = g_vec
        vel = 0.0
        if len(self.g_hist) > 0:
            last_g_x, last_g_y = self.g_hist[-1]
            vel = ((g_x - last_g_x) ** 2 + (g_y - last_g_y) ** 2) ** 0.5
        v_thresh_low, v_thresh_high = 0.02, 0.15  # these might need tuning
        if vel < v_thresh_low:
            self.low_v_count += 1
        else:

            self.low_v_count = 0
        if self.low_v_count > 2:
            self.g_state = 'FIXATION'
        elif vel > v_thresh_high:
            self.g_state = 'SACCADE'


        else:
            self.g_state = 'ERRATIC'
        self.g_hist.append((g_x, g_y))

        # update the personal baseline
        if h_score > 90 and self.g_state == 'FIXATION':
            self.good_gaze_devs.append(g_dev)
        g_score = 0.0
        if self.g_state == 'ERRATIC':
            g_score = 20.0
        elif self.g_state == 'SACCADE':
            g_score = 95.0
        elif self.g_state == 'FIXATION':
            if len(self.good_gaze_devs) > 10:
                base_dev = np.mean(self.good_gaze_devs)
                base_std = np.std(self.good_gaze_devs)
                dist = abs(g_dev - base_dev)
                # penalty based on how many std deviations away from personal norm
                pen = dist / (4 * base_std + 0.01)
                g_score = 100 * (1.0 - min(pen, 1.0))
            else:
                g_score = 100.0  # not enough data for a baseline yet, assume its good

        #    doesnt tank when u just glance at something for a second
        looking_away = h_score < 85 or g_score < 85

        if looking_away and not self.is_off:
            self.is_off = True
            self.off_target_t = ts
        elif not looking_away:
            self.is_off = False
        if self.is_off and (ts - self.off_target_t < 2.0):
            h_score = 100.0
            g_score = 100.0

        # main score
        base_score = h_score * (g_score / 100.0)
        fatigue_mult = 1.0 - blinks['perclos']
        # extra p for a recent blink n dur
        if self.durs and self.durs[-1] > 0.45 and (ts - self.b_times[-1] < 3.0):
            fatigue_mult *= 0.5
        raw_att = base_score * fatigue_mult

        #smooth it out
        self.smooth_att = (self.smooth_att * 0.92) + (raw_att * 0.08)

        return {
            'attention': self.smooth_att,
            'blinks': blinks,
            'gaze': (g_x, g_y, g_dev),
            'gaze_state': self.g_state
        }

    # --------- roi exclusion ---------------------------------
    @staticmethod
    def get_excluded_rois(blend):
       ##frame is mirrored so swapped the stuff
        excluded = []
        f_thresh = 0.3
        cheek_thresh = 0.5
        # forehead exclusion when brows go up
        if (blend[3] > f_thresh and blend[4] > f_thresh) or blend[2] > f_thresh:
            excluded.append("forehead")
        # right side cheek detection
        if blend[7] > cheek_thresh or blend[5] > cheek_thresh or blend[23] > cheek_thresh:
            excluded.append("left_cheek")
        # left side
        if blend[6] > cheek_thresh or blend[5] > cheek_thresh or blend[22] > cheek_thresh:
            excluded.append("right_cheek")

        return excluded