# Vision Transformer Video Anomaly Dashboard Launcher (PowerShell)
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Starting ViT Video Anomaly Operations Dashboard..." -ForegroundColor Green
Write-Host "Target URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\streamlit.exe" run app/streamlit_app.py --server.port 8501
