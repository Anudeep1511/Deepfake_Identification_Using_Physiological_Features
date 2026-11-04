"""DeepDetect - weighted multi-modal feature fusion.

This module is intentionally a rule-based feature-fusion model, not a trained
machine-learning classifier. It combines three modality scores:
    pulse/rPPG 45%, blink/EAR 30%, micro-movement 25%.
"""

import math


class FeatureFusionClassifier:
    def __init__(self, pulse_weight=0.45, blink_weight=0.30,
                 motion_weight=0.25, threshold=0.50):
        total = pulse_weight + blink_weight + motion_weight
        self.w_pulse = pulse_weight / total
        self.w_blink = blink_weight / total
        self.w_motion = motion_weight / total
        self.threshold = threshold

    @staticmethod
    def _clip(value, low=0.0, high=1.0):
        return max(low, min(high, float(value)))

    @staticmethod
    def _sigmoid(x):
        x = max(-15.0, min(15.0, float(x)))
        return 1.0 / (1.0 + math.exp(-x))

    def score_pulse_modality(self, snr_db, hr_bpm, psd_peak_prominence=0.7):
        """Score rPPG physiological consistency from 0 to 1."""
        snr_score = self._sigmoid((float(snr_db) - 3.5) * 0.45)

        if 55.0 <= float(hr_bpm) <= 105.0:
            hr_score = 1.0
        elif 45.0 <= float(hr_bpm) < 55.0 or 105.0 < float(hr_bpm) <= 125.0:
            hr_score = 0.65
        else:
            hr_score = 0.20

        prominence_score = self._clip(psd_peak_prominence, 0.1, 1.0)
        return round(0.55 * snr_score + 0.30 * hr_score + 0.15 * prominence_score, 4)

    def score_blink_modality(self, blink_count, blink_rate_bpm,
                             avg_duration_ms, blink_anomaly_score,
                             blink_valid=True):
        """Score EAR/blink naturalness from 0 to 1."""
        if not blink_valid:
            return 0.50

        base_naturalness = 1.0 - self._clip(blink_anomaly_score)
        rate = float(blink_rate_bpm)
        duration = float(avg_duration_ms)

        if 8.0 <= rate <= 30.0:
            rate_score = 1.0
        elif rate == 0.0:
            rate_score = 0.20
        elif rate < 8.0:
            rate_score = 0.55
        else:
            rate_score = 0.40

        if 100.0 <= duration <= 400.0:
            duration_score = 1.0
        elif duration == 0.0:
            duration_score = 0.30
        elif duration < 90.0:
            duration_score = 0.40
        else:
            duration_score = 0.60

        return round(0.50 * base_naturalness + 0.30 * rate_score + 0.20 * duration_score, 4)

    def score_motion_modality(self, coherence_score, jitter_index, motion_valid=True):
        """Score temporal facial-motion stability from 0 to 1."""
        if not motion_valid:
            return 0.50

        c_score = self._clip(coherence_score)
        jitter = max(0.0, float(jitter_index))

        if jitter <= 0.025:
            j_score = 1.0
        elif jitter <= 0.06:
            j_score = 0.70
        else:
            j_score = max(0.10, 1.0 - jitter * 8.0)

        return round(0.60 * c_score + 0.40 * j_score, 4)

    def classify(self, pulse_metrics, blink_metrics, motion_metrics,
                 pulse_valid=True, blink_valid=True, motion_valid=True):
        """Combine modality scores into an authenticity index.

        The returned synthetic_risk_score is an index derived from the fusion
        score, not a calibrated probability from a trained classifier.
        """
        s_pulse = self.score_pulse_modality(
            pulse_metrics.get("snr_db", 0.0),
            pulse_metrics.get("hr_bpm", 0.0),
            pulse_metrics.get("psd_peak_prominence", 0.1),
        ) if pulse_valid else 0.50

        s_blink = self.score_blink_modality(
            blink_metrics.get("blink_count", 0),
            blink_metrics.get("blink_rate_bpm", 0.0),
            blink_metrics.get("avg_duration_ms", 0.0),
            blink_metrics.get("anomaly_score", 0.5),
            blink_valid=blink_valid,
        )

        s_motion = self.score_motion_modality(
            motion_metrics.get("coherence_score", 0.5),
            motion_metrics.get("jitter_index", 0.04),
            motion_valid=motion_valid,
        )

        fusion_score = (
            self.w_pulse * s_pulse
            + self.w_blink * s_blink
            + self.w_motion * s_motion
        )
        authenticity_index = round(fusion_score * 100.0, 1)
        synthetic_risk_score = round((1.0 - fusion_score) * 100.0, 1)
        is_synthetic = synthetic_risk_score >= self.threshold * 100.0
        verdict = "SYNTHETIC / MANIPULATED" if is_synthetic else "AUTHENTIC HUMAN"

        findings = []
        if not pulse_valid:
            findings.append("Insufficient rPPG evidence")
        elif s_pulse < 0.45:
            findings.append("Weak or incoherent cardiovascular pulse")
        else:
            findings.append("Coherent rPPG pulse detected")

        if not blink_valid:
            findings.append("Blink analysis unavailable or insufficient")
        elif s_blink < 0.45:
            findings.append("Abnormal blink dynamics")
        else:
            findings.append("Natural-looking blink dynamics")

        if not motion_valid:
            findings.append("Motion analysis unavailable or insufficient")
        elif s_motion < 0.45:
            findings.append("Temporal facial-motion instability")
        else:
            findings.append("Stable temporal facial motion")

        return {
            "verdict": verdict,
            "is_synthetic": is_synthetic,
            "authenticity_index": authenticity_index,
            "synthetic_risk_score": synthetic_risk_score,
            "modality_scores": {
                "pulse_coherence": round(s_pulse * 100.0, 1),
                "blink_naturalness": round(s_blink * 100.0, 1),
                "motion_stability": round(s_motion * 100.0, 1),
            },
            "fusion_weights": {
                "pulse_weight": round(self.w_pulse, 2),
                "blink_weight": round(self.w_blink, 2),
                "motion_weight": round(self.w_motion, 2),
            },
            "rationale": "; ".join(findings),
        }