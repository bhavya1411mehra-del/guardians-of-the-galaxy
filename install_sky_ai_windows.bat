@echo off
cd /d "%~dp0"
echo AIRFECTION SKY AI - installing optional real CLIP model dependencies...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m pip install -r requirements-vision.txt
) else (
  py -m pip install -r requirements-vision.txt
)
echo.
echo Setup complete if no errors appear above.
echo The FIRST photo check downloads the CLIP model from Hugging Face.
pause
