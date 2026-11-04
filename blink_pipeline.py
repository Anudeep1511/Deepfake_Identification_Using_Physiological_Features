"""DeepDetect blink/EAR and facial micro-movement analysis."""

import numpy as np


class BlinkDetector:
    def __init__(self, ear_threshold=0.21, consecutive_frames=2, fps=30.0):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.fps = float(fps)

    @staticmethod
    def calculate_ear(eye_landmarks):
        pts = np.asarray(eye_landmarks, dtype=float)
        if len(pts) < 6:
            return 0.30
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        h = np.linalg.norm(pts[0] - pts[3])
        if h < 1e-6:
            return 0.30
        return float((v1 + v2) / (2.0 * h))

    def analyze_ear_sequence(self, ear_series):
        values = np.asarray(ear_series, dtype=float)
        values = values[np.isfinite(values)]
        duration_sec = len(values) / self.fps if self.fps > 0 else 0.0

        if len(values) < 30 or duration_sec < 1.0:
            return {
                "valid": False,
                "blink_count": 0,
                "blink_rate_bpm": 0.0,
                "avg_duration_ms": 0.0,
                "anomaly_score": 0.5,
                "diagnosis": "Insufficient temporal EAR data for blink validation",
            }

        is_closed = values < self.ear_threshold
        blink_runs = []
        run = 0
        for closed in is_closed:
            if closed:
                run += 1
            elif run >= self.consecutive_frames:
                blink_runs.append(run)
                run = 0
            else:
                run = 0
        if run >= self.consecutive_frames:
            blink_runs.append(run)

        count = len(blink_runs)
        rate = count / duration_sec * 60.0
        durations = [(n / self.fps) * 1000.0 for n in blink_runs]
        avg_duration = float(np.mean(durations)) if durations else 0.0

        anomaly = 0.0
        reasons = []
        if duration_sec >= 4.0:
            if count == 0:
                anomaly += 0.45
                reasons.append("No blink events detected")
            elif rate < 6:
                anomaly += 0.25
                reasons.append("Very low blink frequency")
            elif rate > 40:
                anomaly += 0.35
                reasons.append("Very high blink frequency")

            if durations:
                short_ratio = sum(d < 80 for d in durations) / len(durations)
                if short_ratio > 0.4:
                    anomaly += 0.30
                    reasons.append("High proportion of very short closures")
                if any(d > 500 for d in durations):
                    anomaly += 0.20
                    reasons.append("Prolonged eyelid closure detected")

        anomaly = min(1.0, max(0.05, anomaly))
        return {
            "valid": True,
            "blink_count": count,
            "blink_rate_bpm": round(float(rate), 1),
            "avg_duration_ms": round(avg_duration, 1),
            "anomaly_score": round(float(anomaly), 3),
            "diagnosis": "; ".join(reasons) if reasons else "Natural-looking blink pattern",
        }


class MicroMovementAnalyzer:
    def __init__(self, fps=30.0):
        self.fps = float(fps)

    def compute_temporal_coherence(self, motion_vectors):
        values = np.asarray(motion_vectors, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 5:
            return {
                "valid": False,
                "jitter_index": 0.0,
                "coherence_score": 0.5,
                "is_natural": False,
            }

        diffs = np.diff(values)
        jitter = float(np.std(diffs))
        too_stiff = jitter < 0.001
        too_jittery = jitter > 0.08
        coherence = 0.4 if too_stiff else 1.0 - min(1.0, jitter * 8.0)

        return {
            "valid": True,
            "jitter_index": round(jitter, 4),
            "coherence_score": round(float(coherence), 3),
            "is_natural": not (too_stiff or too_jittery),
        }