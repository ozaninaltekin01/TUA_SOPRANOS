"""
start_dev.py — TUA SOPRANOS Tek Komutla Canlı Geliştirme Başlatıcı
Hem FastAPI backend'i hem de Vite React frontend'i tek komutla çalıştırır.
"""
import os
import sys
import subprocess
import time

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "tua_sopranos1")
UI_DIR = os.path.join(ROOT_DIR, "tua_sopranos_ui")

print("=" * 65)
print("  🚀 TUA SOPRANOS — SISTEM BASLATICI")
print("=" * 65)
print(f"📁 Backend : {BACKEND_DIR}")
print(f"📁 Frontend: {UI_DIR}")
print("\n📡 1/2: FastAPI Backend başlatılıyor (http://localhost:8000)...")

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

backend_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
    cwd=BACKEND_DIR,
    env=env,
)

time.sleep(2)
print("💻 2/2: React UI başlatılıyor (http://localhost:5173)...")

ui_proc = subprocess.Popen(
    ["npm", "run", "dev"],
    cwd=UI_DIR,
    shell=True,
)

print("\n" + "=" * 65)
print("  ✅ SISTEM AKTIF!")
print("  🔗 React Arayüz : http://localhost:5173")
print("  🔗 FastAPI Docs : http://localhost:8000/docs")
print("  🔗 API Health   : http://localhost:8000/api/status")
print("  Kapatmak için CTRL+C tuşlarına basabilirsiniz.")
print("=" * 65 + "\n")

try:
    backend_proc.wait()
    ui_proc.wait()
except KeyboardInterrupt:
    print("\n🛑 Sistem kapatılıyor...")
    backend_proc.terminate()
    ui_proc.terminate()
    print("👋 Kapatıldı.")
