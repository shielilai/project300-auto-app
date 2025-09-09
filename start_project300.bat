@echo off
REM === Project300 Auto POC Launcher ===
SETLOCAL

REM Go to repo root (this batch is placed in repo root)
cd /d %~dp0

REM Check if venv exists
IF NOT EXIST .venv (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate

REM Upgrade pip etc once
python -m pip install --upgrade pip setuptools wheel

REM Install requirements
pip install -r requirements.txt

REM Launch Streamlit app
cd day3
streamlit run app.py

ENDLOCAL