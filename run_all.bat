@echo off
chcp 65001 >nul
echo ===================================================
echo [TUA SOPRANOS] Tum Sistem (Backend + UI) Baslatiliyor...
echo ===================================================
start "TUA SOPRANOS - Backend API (Port 8000)" cmd /k "%~dp0run_backend.bat"
timeout /t 2 /nobreak >nul
start "TUA SOPRANOS - React UI (Port 5173)" cmd /k "%~dp0run_ui.bat"
