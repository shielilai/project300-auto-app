# Project 300 — Automotive Streamlit Bundle

This bundle contains a **Streamlit app** and a **Jupyter notebook** for your automotive data‑cleansing + FastSale prediction POC.

## What's inside (day3/)
- `app.py` — Streamlit app (upload → validate → auto‑fix → model → download)
- `start_streamlit.bat` — one‑click launcher (Windows)
- `auto_sales_sample.csv` — sample dataset
- `(add your notebook here)` — working notebook
- `requirements.txt` — Python libs for both Streamlit/notebook

## Quickstart (Windows)
1. Create folder `C:\project300\day3`.
2. Unzip these files **into** `C:\project300\day3`.
3. Open Command Prompt:
   ```bat
   cd C:\project300\day3
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bat
   start_streamlit.bat
   ```
   or
   ```bat
   streamlit run app.py
   ```

## Jupyter Notebook
If you prefer Jupyter:
```bat
cd C:\project300\day3
jupyter notebook
```
Open `Project300_Auto.ipynb` and run cells top → bottom.

## Notes
- The app will attempt to load `auto_sales_sample.csv` by default;
  you can also upload your own automotive CSV.
- The model is a **simple baseline** (RandomForest) to illustrate feasibility.