@echo off
REM Vision Transformer Video Anomaly Dashboard Launcher
echo ==========================================================
echo Starting ViT Video Anomaly Operations Dashboard...
echo Local URL: http://localhost:8501
echo ==========================================================
".\.venv\Scripts\streamlit.exe" run app/streamlit_app.py --server.port 8501 --server.headless false
pause
