@echo off
setlocal

REM --- Streamlit launcher for Project300 Automotive ---
set TARGET=C:\project300\day3

if not exist "%TARGET%" (
  echo [Error] Folder not found: %TARGET%
  pause
  exit /b 1
)

cd /d "%TARGET%"
echo Starting Streamlit app in %CD% ...
streamlit run app.py