@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project Python environment is missing. Run the setup commands in README.md first.
  pause
  exit /b 1
)
echo Starting SarcasmLens backend. Keep this window open.
echo Loading the model may take a little time. Wait for Application startup complete.
".venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
  echo.
  echo Backend failed to start. See the error above. If port 8000 is in use, another backend is already running.
  pause
)
