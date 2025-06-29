from collections import deque
import numpy as np

class Expression:
    def __init__(self, threshold=0.5, window_seconds=61):
        self.threshold = threshold
        self.window_seconds = window_seconds

        # ---------- blink tracking ----------
        self.timestamps = deque()
        self.blink_signals = deque()
        self.blink_events = deque()
        self.last_blink_val = 0

        # --- adaptive-baseline parameters ---
        self.ema_alpha = 0.001                    # very slow drift  (~17 min half-life)
        self.blink_rate_buffer = deque(maxlen=120)  # ≈2-min history of blink-rates

        # ---------- cognitive buffers ----------
        self.cognitive_sample_count = 0
        buffer_size = min(300, int(window_seconds * 5))

        self.brow_buffer = deque(maxlen=buffer_size)
        self.eye_strain_buffer = deque(maxlen=buffer_size)
        self.gaze_buffer = deque(maxlen=buffer_size)
        self.micro_buffer = deque(maxlen=buffer_size)
        self.eye_opening_buffer = deque(maxlen=buffer_size)

        self.baseline_samples = min(150, int(window_seconds * 2.5))
        self.baseline_established = False
        self.baseline_stats = {
            'brow_mean': 0, 'brow_std': 0.1,
            'eye_strain_mean': 0, 'eye_strain_std': 0.1,
            'gaze_mean': 0, 'gaze_std': 0.1,
            'micro_mean': 0, 'micro_std': 0.1,
            'eye_opening_mean': 0, 'eye_opening_std': 0.1
        }

    def get_cognitive(self, blendshapes, timestamp):
        if len(blendshapes) < 52:
            return self._get_default_response()

        # ---- blink detection ----------------------------------------
        blink_val = max(blendshapes[8], blendshapes[9])  # left / right blink blendshapes
        self.timestamps.append(timestamp)
        self.blink_signals.append(blink_val)

        # trim stale entries
        while self.timestamps and timestamp - self.timestamps[0] > self.window_seconds:
            self.timestamps.popleft()
            self.blink_signals.popleft()
        while self.blink_events and timestamp - self.blink_events[0] > self.window_seconds:
            self.blink_events.popleft()

        blink_detected = self.last_blink_val >= self.threshold > blink_val
        if blink_detected:
            self.blink_events.append(timestamp)
        self.last_blink_val = blink_val

        # ---- blink metrics ------------------------------------------
        duration = (self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 1.0
        blink_count = len(self.blink_events)
        blink_rate_per_window = (blink_count / duration) * self.window_seconds if duration > 0 else 0.0
        blink_rate_per_minute = blink_rate_per_window * (60 / self.window_seconds)

        # keep rolling history for adaptive baseline
        self.blink_rate_buffer.append(blink_rate_per_minute)

        ibi = np.diff(np.array(self.blink_events)) if blink_count > 1 else np.array([])
        mean_ibi = np.mean(ibi) if ibi.size else 0.0
        sd_ibi = np.std(ibi) if ibi.size else 0.0
        rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2)) if ibi.size > 2 else 0.0
        cv_ibi = sd_ibi / mean_ibi if mean_ibi > 0 else 0.0

        blink_durations = self._estimate_blink_durations()
        mean_blink_duration = np.mean(blink_durations) if blink_durations else 0.0
        suppression_periods = ibi[ibi > 2 * mean_ibi] if ibi.size else np.array([])
        mean_suppression = np.mean(suppression_periods) if suppression_periods.size else 0.0

        blink_metrics = {
            "blink_pm": blink_rate_per_minute,
            "blink_count": blink_count,
            "cv_ibi": cv_ibi,
            "rmssd": rmssd
        }
        cognitive_state = self._update_cognitive_state(blendshapes, timestamp, blink_metrics)

        return {
            "blink": {
                "blink_pm": blink_rate_per_minute,
                "blink_count": blink_count,
                "mean_ibi": mean_ibi,
                "sd_ibi": sd_ibi,
                "rmssd": rmssd,
                "cv_ibi": cv_ibi,
                "mean_blink_duration": mean_blink_duration,
                "mean_suppression_period": mean_suppression,
                "blink_detected": blink_detected
            },
            "cognitive": cognitive_state
        }

    # internal helpers -------------------------------------------------
    def _update_cognitive_state(self, blendshapes, timestamp, blink_metrics):
        default_cognitive = {
            'status': 'initializing',
            'progress': 0.0,
            'stress_score': 0.0,
            'attention_score': 0.0,
            'stress_level': 'UNKNOWN',
            'attention_level': 'UNKNOWN',
            'details': {
                'brow_tension': 0.0,
                'eye_strain': 0.0,
                'gaze_movement': 0.0,
                'eye_opening': 0.0,
                'blink_rate': blink_metrics.get('blink_pm', 0),
                'samples': 0
            }
        }

        # sample every 5th frame
        if self.cognitive_sample_count % 5 != 0:
            self.cognitive_sample_count += 1
            return getattr(self, '_last_cognitive_state', default_cognitive)

        if len(blendshapes) < 52:
            default_cognitive['status'] = 'insufficient_data'
            self._last_cognitive_state = default_cognitive
            return default_cognitive

        # ---------- derive metrics ----------
        brow_tension = (blendshapes[0] + blendshapes[1] + blendshapes[2]) / 3
        eye_strain = (blendshapes[6] + blendshapes[7] + blendshapes[18] + blendshapes[19]) / 4
        gaze_movement = np.mean(blendshapes[10:18])
        micro_movement = (blendshapes[5] + blendshapes[27] + blendshapes[28] +
                          blendshapes[49] + blendshapes[50]) / 5
        eye_opening = (blendshapes[20] + blendshapes[21]) / 2

        # store
        self.brow_buffer.append(brow_tension)
        self.eye_strain_buffer.append(eye_strain)
        self.gaze_buffer.append(gaze_movement)
        self.micro_buffer.append(micro_movement)
        self.eye_opening_buffer.append(eye_opening)

        self.cognitive_sample_count += 1

        # progress
        current_samples = len(self.brow_buffer)
        progress = min(current_samples / self.baseline_samples, 1.0)

        default_cognitive['progress'] = progress
        default_cognitive['details'].update(
            brow_tension=float(brow_tension),
            eye_strain=float(eye_strain),
            gaze_movement=float(gaze_movement),
            eye_opening=float(eye_opening),
            samples=current_samples
        )

        # establish baseline
        if not self.baseline_established:
            if current_samples >= self.baseline_samples:
                self._calculate_baseline_stats()
                self.baseline_established = True
                default_cognitive['status'] = 'baseline_complete'
            else:
                default_cognitive['status'] = 'establishing_baseline'
            self._last_cognitive_state = default_cognitive
            return default_cognitive

        # ---------- slow EMA drift of baseline (Upgrade A) ----------
        self._ema_update('brow_mean',        'brow_std',        brow_tension)
        self._ema_update('eye_strain_mean',  'eye_strain_std',  eye_strain)
        self._ema_update('gaze_mean',        'gaze_std',        gaze_movement)
        self._ema_update('micro_mean',       'micro_std',       micro_movement)
        self._ema_update('eye_opening_mean', 'eye_opening_std', eye_opening)

        # analyze
        try:
            result = self._analyze_cognitive_metrics(blink_metrics)
            if result:
                result['progress'] = 1.0
                result['details'].update(
                    brow_tension=float(brow_tension),
                    eye_strain=float(eye_strain),
                    gaze_movement=float(gaze_movement),
                    eye_opening=float(eye_opening),
                    samples=current_samples
                )
                self._last_cognitive_state = result
                return result
            default_cognitive['status'] = 'analysis_failed'
        except Exception:
            default_cognitive['status'] = 'error'

        default_cognitive['progress'] = progress
        self._last_cognitive_state = default_cognitive
        return default_cognitive

    # --------- baseline helpers --------------------------------------
    def _calculate_baseline_stats(self):
        def robust_stats(deq):
            if not deq:
                return 0.0, 0.1
            arr = np.array(deq)
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            std_est = max(mad * 1.4826, 0.01)
            return float(med), float(std_est)

        for key_base, buf in [('brow', self.brow_buffer),
                              ('eye_strain', self.eye_strain_buffer),
                              ('gaze', self.gaze_buffer),
                              ('micro', self.micro_buffer),
                              ('eye_opening', self.eye_opening_buffer)]:
            mean_key = f'{key_base}_mean'
            std_key = f'{key_base}_std'
            self.baseline_stats[mean_key], self.baseline_stats[std_key] = robust_stats(buf)

    # ema base line
    def _ema_update(self, mean_key, std_key, x):
        a = self.ema_alpha
        mu = self.baseline_stats[mean_key]
        sig = self.baseline_stats[std_key]
        mu_new = (1 - a) * mu + a * x
        sig_new = (1 - a) * sig + a * abs(x - mu_new)
        self.baseline_stats[mean_key] = mu_new
        self.baseline_stats[std_key] = max(sig_new, 0.01)

    # blink baseline
    def _get_blink_baseline(self):
        if len(self.blink_rate_buffer) >= 5:
            return float(np.median(self.blink_rate_buffer))
        return 15.0

    # metric analysis
    def _analyze_cognitive_metrics(self, blink_metrics):
        window = min(10, len(self.brow_buffer))
        if window < 3:
            return {
                'status': 'insufficient_samples',
                'progress': 1.0,
                'stress_score': 0.0,
                'attention_score': 0.0,
                'stress_level': 'UNKNOWN',
                'attention_level': 'UNKNOWN',
                'details': {
                    'brow_z': 0.0, 'eye_strain_z': 0.0,
                    'gaze_z': 0.0, 'eye_opening_z': 0.0,
                    'blink_rate': blink_metrics.get('blink_pm', 0),
                    'samples': len(self.brow_buffer)
                }
            }

        # recent averages
        brow_avg = np.mean(list(self.brow_buffer)[-window:])
        eye_avg = np.mean(list(self.eye_strain_buffer)[-window:])
        gaze_avg = np.mean(list(self.gaze_buffer)[-window:])
        micro_avg = np.mean(list(self.micro_buffer)[-window:])
        open_avg = np.mean(list(self.eye_opening_buffer)[-window:])

        # z-scores
        def z(v, m, s):
            return np.clip((v - m) / s, -3, 3)

        brow_z = z(brow_avg,  self.baseline_stats['brow_mean'],        self.baseline_stats['brow_std'])
        eye_z  = z(eye_avg,   self.baseline_stats['eye_strain_mean'],  self.baseline_stats['eye_strain_std'])
        gaze_z = z(gaze_avg,  self.baseline_stats['gaze_mean'],        self.baseline_stats['gaze_std'])
        micro_z = z(micro_avg, self.baseline_stats['micro_mean'],      self.baseline_stats['micro_std'])
        open_z = z(open_avg,  self.baseline_stats['eye_opening_mean'], self.baseline_stats['eye_opening_std'])

        # adaptive blink reference
        blink_rate = blink_metrics.get('blink_pm', 15)
        blink_opt = self._get_blink_baseline()
        blink_cv = blink_metrics.get('cv_ibi', 0.3)

        stress_score = (
            max(0, brow_z) * 0.30 +
            max(0, eye_z) * 0.25 +
            max(0, open_z) * 0.15 +
            abs(blink_rate - blink_opt) / 10 * 0.20 +
            max(0, (blink_cv - 0.3) / 0.3) * 0.10
        )

        attention_score = (
            max(0, -gaze_z) * 0.30 +
            max(0, -micro_z) * 0.20 +
            max(0, open_z) * 0.15 +
            max(0, 1 - abs(blink_rate - blink_opt) / 10) * 0.20 +
            max(0, 1 - blink_cv / 0.6) * 0.15
        )

        def classify(score, th=[0.3, 0.6, 0.8]):
            return ("LOW", "MILD", "MODERATE", "HIGH")[
                sum(score >= np.array(th))
            ]

        return {
            'status': 'active',
            'progress': 1.0,
            'stress_score': float(np.clip(stress_score, 0, 2)),
            'attention_score': float(np.clip(attention_score, 0, 1)),
            'stress_level': classify(stress_score),
            'attention_level': classify(attention_score, [0.2, 0.5, 0.7]),
            'details': {
                'brow_z': float(brow_z),
                'eye_strain_z': float(eye_z),
                'gaze_z': float(gaze_z),
                'eye_opening_z': float(open_z),
                'blink_rate': float(blink_rate),
                'samples': len(self.brow_buffer)
            }
        }

    # ______________________helpers_______________________________
    def _estimate_blink_durations(self):
        durations = []
        ev = list(self.blink_events)
        for i in range(len(ev) - 1):
            d = ev[i + 1] - ev[i]
            if 0 < d < 1.0:
                durations.append(d)
        return durations

    def _get_default_response(self):
        return {
            "blink": {
                "blink_pm": 0, "blink_count": 0, "mean_ibi": 0, "sd_ibi": 0,
                "rmssd": 0, "cv_ibi": 0, "mean_blink_duration": 0,
                "mean_suppression_period": 0, "blink_detected": False
            },
            "cognitive": {
                "status": "insufficient_blendshapes", "progress": 0.0,
                "stress_score": 0.0, "attention_score": 0.0,
                "stress_level": "UNKNOWN", "attention_level": "UNKNOWN",
                "details": {"samples": 0, "blink_rate": 0}
            }
        }

    # --------- roi exclusion ---------------------------------
    @staticmethod
    def get_excluded_rois(blendshapes):
        if len(blendshapes) < 52:
            return []

        excluded = []
        forehead_threshold = 0.3
        cheek_threshold = 0.7

        if blendshapes[2] > forehead_threshold or blendshapes[3] > forehead_threshold or blendshapes[4] > forehead_threshold:
            excluded.append("forehead")
        if blendshapes[44] > cheek_threshold or blendshapes[5] > cheek_threshold or blendshapes[7] > cheek_threshold:
            excluded.append("right_cheek")
        if blendshapes[43] > cheek_threshold or blendshapes[5] > cheek_threshold or blendshapes[6] > cheek_threshold:
            excluded.append("left_cheek")

        return excluded