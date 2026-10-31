@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title AIRFECTION - Install and Run

echo ========================================================
echo    AIRFECTION - Windows Setup

echo ========================================================
echo.

if not exist "app.py" (
  echo ERROR: app.py was not found in this folder.
  echo Extract the ZIP first, then run setup_windows.bat from inside airfection_website.
  goto :failed
)

set "PYTHON_CMD="
where py >nul 2>&1
if not errorlevel 1 (
  for %%V in (3.12 3.11 3.13) do (
    py -%%V -c "import sys; print(sys.version)" >nul 2>&1
    if not errorlevel 1 (
      set "PYTHON_CMD=py -%%V"
      goto :python_found
    )
  )
)

where python >nul 2>&1
if not errorlevel 1 (
  python -c "import sys; print(sys.version)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :python_found
  )
)

echo ERROR: Could not find a working Python installation.
echo Install Python 3.12 from https://www.python.org/downloads/
echo During installation, check "Add python.exe to PATH" if available.
echo After installation, close and reopen VS Code or Command Prompt.
goto :failed

:python_found
echo Detected Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating a project environment...
  %PYTHON_CMD% -m venv ".venv"
  if errorlevel 1 (
    echo ERROR: Could not create the Python environment.
    echo You may need to reinstall Python with the pip and venv components.
    goto :failed
  )
) else (
  echo [1/3] Using existing project environment.
)

".venv\Scripts\python.exe" --version
if errorlevel 1 (
  echo ERROR: The existing .venv environment cannot run.
  echo In VS Code, delete the .venv folder and try this setup again.
  goto :failed
)

echo.
echo [2/3] Installing the required libraries. This needs internet access.
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 (
  echo.
  echo ERROR: Library installation failed. Read the error above.
  echo Check internet access and the Python version.
  goto :failed
)

echo.
echo [3/3] Starting AIRFECTION...
echo The website should open at http://localhost:8501
".venv\Scripts\python.exe" -m streamlit run "app.py"
if errorlevel 1 (
  echo.
  echo ERROR: AIRFECTION failed to start. Read the error above.
  goto :failed
)

echo.
echo AIRFECTION has stopped. You can close this window.
pause
exit /b 0

:failed
echo.
echo Setup did not complete. Take a screenshot of this window
 echo and send it to ChatGPT so the exact error can be fixed.
echo.
pause
exit /b 1
