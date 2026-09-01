@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
echo ===================================================
echo [TUA SOPRANOS] FastAPI Backend Baslatiliyor...
echo Dokumantasyon: http://localhost:8000/docs
echo API Durum:     http://localhost:8000/api/status
echo ===================================================
cd /d "%~dp0tua_sopranos1"
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
pause
