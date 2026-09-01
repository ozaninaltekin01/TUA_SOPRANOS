@echo off
chcp 65001 >nul
echo ===================================================
echo [TUA SOPRANOS] React UI (Vite) Baslatiliyor...
echo Arayuz: http://localhost:5173
echo ===================================================
cd /d "%~dp0tua_sopranos_ui"
npm run dev
pause
