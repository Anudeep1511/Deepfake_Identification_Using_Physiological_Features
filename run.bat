@echo off
title DeepDetect
echo ========================================
echo DeepDetect - Physiological Feature Analysis
echo ========================================
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Check your Python version and internet connection.
  pause
  exit /b 1
)
start "" http://localhost:5000
python app.py
pause