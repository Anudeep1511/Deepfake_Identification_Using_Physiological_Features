"""DeepDetect - Flask backend for physiological deepfake analysis.

Pipeline:
Face detection -> physiological ROI -> rPPG -> EAR/blink -> micro-movement
-> weighted feature fusion -> authenticity assessment.

Important: this backend uses a deterministic feature-fusion model. It is not
a trained deep-learning classifier and does not claim calibrated probabilities.
"""

import math
import os
import tempfile
import numpy as np

try:
    import cv2
    CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH) if os.path.exists(CASCADE_PATH) else None
    CV2_AVAILABLE = cv2 is not None and (face_cascade is not None and not face_cascade.empty())
except Exception:
    cv2 = None
    face_cascade = None
    CV2_AVAILABLE = False

face_mesh_detector = None
face_landmarker = None
MEDIAPIPE_AVAILABLE = False

try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        MEDIAPIPE_AVAILABLE = True
    else:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        MEDIAPIPE_AVAILABLE = True
except Exception as e:
    mp = None
    face_mesh_detector = None
    face_landmarker = None
    MEDIAPIPE_AVAILABLE = False

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from rppg_pipeline import RPPGExtractor
from blink_pipeline import BlinkDetector, MicroMovementAnalyzer
from fusion_classifier import FeatureFusionClassifier
from pathlib import Path

# Dataset location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "REAL_FAKE_DATASET"

REAL_PATH = DATASET_PATH / "real"
FAKE_PATH = DATASET_PATH / "fake"
app = Flask(__name__, static_folder=".", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
CORS(app)

MAX_ANALYSIS_SECONDS = 12
MIN_ANALYSIS_FRAMES = 45


def make_engines(fps):
    return (
        RPPGExtractor(fps=fps),
        BlinkDetector(fps=fps),
        MicroMovementAnalyzer(fps=fps),
    )


fusion_classifier = FeatureFusionClassifier(
    pulse_weight=0.45,
    blink_weight=0.30,
    motion_weight=0.25,
)


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"success": False, "error": "Video exceeds the 50 MB upload limit."}), 413


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "status": "online",
        "service": "DeepDetect",
        "version": "3.0.0",
        "architecture": "Multi-Modal Physiological Feature Fusion",
        "features_supported": [
            "Face detection and physiological ROI extraction",
            "rPPG BVP extraction using CHROM/POS",
            "FFT heart-rate and SNR analysis",
            "EAR blink dynamics",
            "Facial micro-movement temporal coherence",
            "Weighted feature fusion: 45% pulse / 30% blink / 25% motion",
        ],
        "opencv_available": CV2_AVAILABLE,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
    })


@app.route("/api/samples", methods=["GET"])
def samples():
    """Return clearly labelled demo test cases, not research ground truth."""
    return jsonify({"samples": [
        {
            "id": "test-auth-01",
            "title": "Test Case 01 — Authentic Human",
            "type": "real",
            "is_simulated": True,
            "description": "Reference profile with natural pulse and blinking patterns",
        },
        {
            "id": "test-fake-02",
            "title": "Test Case 02 — FaceSwap Manipulation",
            "type": "fake",
            "is_simulated": True,
            "description": "Synthetic profile with inconsistent physiological pulse characteristics",
        },
        {
            "id": "test-fake-03",
            "title": "Test Case 03 — Diffusion-Based Manipulation",
            "type": "fake",
            "is_simulated": True,
            "description": "Synthetic profile with abnormal temporal micro-movement patterns",
        },
    ]})


def _safe_face_box(box, width, height):
    x, y, w, h = [int(v) for v in box]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def _mediapipe_ear_and_box(frame_rgb):
    if not MEDIAPIPE_AVAILABLE:
        raise ValueError("MediaPipe is required for geometric EAR blink analysis. Install requirements.txt and restart the backend.")

    landmarks = None
    h, w = frame_rgb.shape[:2]

    if face_mesh_detector is not None:
        result = face_mesh_detector.process(frame_rgb)
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
    elif face_landmarker is not None:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = face_landmarker.detect(mp_img)
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

    if not landmarks:
        return None, None

    left_ids = [33, 160, 158, 133, 153, 144]
    right_ids = [362, 385, 387, 263, 373, 380]
    left = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in left_ids])
    right = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in right_ids])
    left_ear = BlinkDetector.calculate_ear(left)
    right_ear = BlinkDetector.calculate_ear(right)
    ear = float((left_ear + right_ear) / 2.0)

    xs = np.array([p.x * w for p in landmarks])
    ys = np.array([p.y * h for p in landmarks])
    box = (
        int(xs.min()), int(ys.min()),
        int(xs.max() - xs.min()), int(ys.max() - ys.min())
    )
    return ear, _safe_face_box(box, w, h)


def _fallback_face_and_ear(gray, width, height):
    if face_cascade is None:
        return None, None
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None, None
    # Largest detected face is used.
    face = max(faces, key=lambda b: int(b[2]) * int(b[3]))
    x, y, w, h = _safe_face_box(face, width, height)

    # Haar can provide a face ROI if Face Mesh misses a frame, but it cannot
    # provide geometric EAR. Never infer EAR from pixel contrast.
    return (x, y, w, h), None


def _normalise_signal(signal_data):
    arr = np.asarray(signal_data, dtype=float)
    return ((arr - np.mean(arr)) / (np.std(arr) + 1e-8)).tolist()


def _analyse_video(video_path):
    if not CV2_AVAILABLE or cv2 is None:
        raise ValueError("OpenCV is unavailable. Install the project requirements.")
    if not MEDIAPIPE_AVAILABLE or (face_mesh_detector is None and face_landmarker is None):
        raise ValueError("MediaPipe is required for genuine geometric EAR analysis. Run: python -m pip install -r requirements.txt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps and source_fps > 1 else 30.0
    fps = min(float(fps), 60.0)
    rppg, blink, motion = make_engines(fps)

    rgb_series = []
    ear_series = []
    motion_values = []
    face_frames = 0
    frame_count = 0
    prev_gray = None
    max_frames = max(MIN_ANALYSIS_FRAMES, int(fps * MAX_ANALYSIS_SECONDS))

    try:
        while cap.isOpened() and frame_count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            h, w = frame.shape[:2]
            scale = min(1.0, 640.0 / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                h, w = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ear, face_box = _mediapipe_ear_and_box(rgb)
            if face_box is None:
                face_box, fallback_ear = _fallback_face_and_ear(gray, w, h)
                ear = fallback_ear

            if face_box is None:
                # Do not fabricate a face/ROI. Missing-face frames are excluded.
                prev_gray = gray
                continue

            face_frames += 1
            if ear is not None:
                ear_series.append(float(ear))

            rgb_series.append(rppg.extract_skin_rois(frame, face_box))

            # Restrict motion measurement to the detected face region.
            x, y, fw, fh = face_box
            face_gray = gray[y:y+fh, x:x+fw]
            if prev_gray is not None and face_gray.size and prev_gray.shape == gray.shape:
                previous_face = prev_gray[y:y+fh, x:x+fw]
                if previous_face.shape == face_gray.shape:
                    motion_values.append(float(np.mean(cv2.absdiff(face_gray, previous_face)) / 255.0))
            prev_gray = gray
    finally:
        cap.release()

    duration = frame_count / fps if fps else 0.0
    if frame_count < MIN_ANALYSIS_FRAMES:
        raise ValueError(f"Video is too short for reliable analysis. At least {MIN_ANALYSIS_FRAMES} frames are required.")
    if len(rgb_series) < MIN_ANALYSIS_FRAMES:
        raise ValueError("A face could not be detected in enough frames for physiological analysis.")

    rgb_arr = np.asarray(rgb_series, dtype=float)[:, ::-1]  # BGR -> RGB
    # Combine CHROM and POS rPPG estimates after normalising their scales.
    chrom_signal = np.asarray(rppg.chrom_rppg(rgb_arr), dtype=float)
    pos_signal = np.asarray(rppg.pos_rppg(rgb_arr), dtype=float)
    if (not np.all(np.isfinite(chrom_signal)) or not np.all(np.isfinite(pos_signal))
            or np.std(chrom_signal) < 1e-7 or np.std(pos_signal) < 1e-7):
        raise ValueError("The facial colour signal is too weak or unstable for rPPG analysis. Try a clearer, well-lit video.")
    chrom_z = (chrom_signal - np.mean(chrom_signal)) / (np.std(chrom_signal) + 1e-8)
    pos_z = (pos_signal - np.mean(pos_signal)) / (np.std(pos_signal) + 1e-8)
    raw_bvp = 0.5 * chrom_z + 0.5 * pos_z
    filtered_bvp = np.asarray(rppg.bandpass_filter(raw_bvp), dtype=float)
    if len(filtered_bvp) < MIN_ANALYSIS_FRAMES or not np.all(np.isfinite(filtered_bvp)) or np.std(filtered_bvp) < 1e-7:
        raise ValueError("The extracted rPPG signal is insufficient. Try a longer, well-lit video with a clearly visible face.")
    hr_bpm, snr_db, freqs, psd = rppg.compute_spectral_hr(filtered_bvp)
    if not np.isfinite(hr_bpm) or not np.isfinite(snr_db) or len(freqs) == 0:
        raise ValueError("Could not identify a reliable pulse peak from the uploaded video.")

    if len(ear_series) < MIN_ANALYSIS_FRAMES:
        raise ValueError("Too few valid MediaPipe eye-landmark frames were detected. Use a front-facing video with both eyes visible.")
    blink_metrics = blink.analyze_ear_sequence(ear_series)
    blink_valid = bool(blink_metrics.get("valid", False))

    motion_metrics = motion.compute_temporal_coherence(motion_values)
    motion_valid = len(motion_values) >= 5 and bool(motion_metrics.get("valid", False))

    pulse_valid = np.isfinite(hr_bpm) and np.isfinite(snr_db) and len(filtered_bvp) >= MIN_ANALYSIS_FRAMES
    peak_prominence = 0.85 if snr_db > 5 else max(0.1, min(1.0, (snr_db + 5.0) / 12.0))

    fusion = fusion_classifier.classify(
        pulse_metrics={"snr_db": snr_db, "hr_bpm": hr_bpm, "psd_peak_prominence": peak_prominence},
        blink_metrics=blink_metrics,
        motion_metrics=motion_metrics,
        pulse_valid=pulse_valid,
        blink_valid=blink_valid,
        motion_valid=motion_valid,
    )

    return {
        "success": True,
        "is_simulated": False,
        "execution_mode": "LIVE_PIPELINE_INFERENCE",
        "label": fusion["verdict"],
        "authenticity_index": fusion["authenticity_index"],
        "synthetic_risk_score": fusion["synthetic_risk_score"],
        "modality_scores": fusion["modality_scores"],
        "fusion_weights": fusion["fusion_weights"],
        "hr_bpm": round(float(hr_bpm), 1),
        "pulse_snr_db": round(float(snr_db), 2),
        "blink_metrics": blink_metrics,
        "motion_coherence": motion_metrics,
        "analysis": {
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 2),
            "frames_read": frame_count,
            "face_frames": face_frames,
            "ear_frames": len(ear_series),
            "motion_frames": len(motion_values),
            "landmark_method": "MediaPipe Face Mesh geometric EAR",
            "pulse_valid": pulse_valid,
            "blink_valid": blink_valid,
            "motion_valid": motion_valid,
        },
        "bvp_signal": [round(float(x), 4) for x in _normalise_signal(filtered_bvp)],
        "ear_signal": [round(float(x), 4) for x in ear_series],
        "timestamps": [round(i / fps, 3) for i in range(len(filtered_bvp))],
        "psd_freqs": [round(float(x), 3) for x in freqs],
        "psd_power": [round(float(x), 5) for x in psd],
        "explanation": fusion["rationale"],
    }


@app.route("/api/analyze", methods=["POST"])
def analyze_video():
    """Analyze an uploaded video through the complete DeepDetect pipeline."""
    video_file = request.files.get("video")
    if video_file is None or not video_file.filename:
        return jsonify({"success": False, "error": "Please upload a video file."}), 400

    suffix = os.path.splitext(video_file.filename)[1].lower() or ".mp4"
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if suffix not in allowed_extensions:
        return jsonify({"success": False, "error": "Unsupported video extension. Use MP4, MOV, AVI, MKV, WebM, or M4V."}), 415
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            temp_path = tmp.name
            video_file.save(temp_path)
        result = _analyse_video(temp_path)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    except Exception as exc:
        app.logger.exception("Video analysis failed")
        return jsonify({"success": False, "error": "Video analysis failed. Check the video format and backend dependencies."}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 64)
    print(" DeepDetect - Multi-Modal Physiological Feature Analysis")
    print(f" Server: http://localhost:{port}")
    print(" Fusion: Pulse 45% + Blink/EAR 30% + Motion 25%")
    print(f" OpenCV: {'Ready' if CV2_AVAILABLE else 'Missing'}")
    print(f" MediaPipe: {'Ready' if MEDIAPIPE_AVAILABLE else 'Missing - upload analysis disabled'}")
    print("=" * 64)
    print("Dataset location:", DATASET_PATH)
    print("Real folder exists:", REAL_PATH.exists())
    print("Fake folder exists:", FAKE_PATH.exists())
    app.run(host="0.0.0.0", port=port, debug=True)