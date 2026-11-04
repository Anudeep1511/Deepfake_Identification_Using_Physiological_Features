# DeepDetect — Physiological Deepfake Analysis Prototype

DeepDetect analyzes an uploaded video using three signal groups:
- **rPPG pulse signal**: CHROM and POS estimates combined for spectral heart-rate/SNR analysis.
- **Blink dynamics**: geometric Eye Aspect Ratio (EAR) from MediaPipe Face Mesh.
- **Facial temporal motion**: frame-to-frame motion variation and coherence.

A rule-based feature-fusion score combines pulse (45%), blink (30%), and motion (25%). The **Authenticity Index** and **Synthetic Risk Score** are heuristic indices, not calibrated probabilities. This project does not currently use a trained deepfake classification neural network, and no benchmark accuracy is claimed.

## Project files
- `index.html` — DeepDetect browser interface
- `app.py` — Flask API and uploaded-video processing
- `rppg_pipeline.py` — CHROM/POS, bandpass filtering, FFT heart-rate/SNR
- `blink_pipeline.py` — geometric EAR blink analysis and motion coherence
- `fusion_classifier.py` — weighted rule-based feature fusion
- `requirements.txt` — Python dependencies
- `run.bat` — Windows setup/launch helper
- `run.sh` — macOS/Linux setup/launch helper

## Requirements
Use Python 3.10 or 3.11 for the most reliable compatibility with MediaPipe. MediaPipe installation depends on Python version and operating system.

## Run on Windows
1. Keep all project files in the same folder.
2. Open a terminal in that folder.
3. Run:
   ```bash
   python -m pip install -r requirements.txt
   python app.py
   ```
4. Open `http://localhost:5000` in your browser. Do not open `index.html` directly, because upload analysis needs Flask.
5. Choose **Upload Video**, select a clip, then click **Analyze Uploaded Video**.

## Run on macOS/Linux
```bash
python3 -m pip install -r requirements.txt
python3 app.py
```
Then visit `http://localhost:5000`.

## Video guidance
Use a short, clear, front-facing video with visible eyes, good lighting, limited camera movement, and a face visible for several seconds. The prototype analyzes up to 12 seconds. Compression, lighting, head movement, occlusion, frame rate, and video length can affect results. If signal quality is insufficient, try a better clip rather than treating the result as evidence of authenticity.

## Test Samples
The three test cards are **descriptive scenarios only**. They are not embedded videos, verified ground-truth examples, or scored benchmark samples. Upload an actual video to run the pipeline.

## API
- `GET /api/status` — backend/dependency status
- `GET /api/samples` — descriptive test-scenario labels
- `POST /api/analyze` — multipart upload with field name `video`

Errors are returned as JSON with `success: false` and an `error` message. Failed uploads do not produce simulated results.

## Limitations
This is an academic prototype, not a forensic guarantee. Physiological features are noisy and may be unreliable in short, compressed, poorly lit, or heavily moving videos. The fusion logic is hand-designed and has not been calibrated or validated against a benchmark dataset. Do not use it as the sole basis for claims about a person or video.
