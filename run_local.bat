@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is not installed.
  echo Run: .\setup_local.bat
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
endlocal
