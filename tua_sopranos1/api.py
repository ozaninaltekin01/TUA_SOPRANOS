"""
api.py — TUA SOPRANOS REST API (FastAPI)
=========================================
React frontend'in baglanacagi HTTP servisi.

Kurulum:
    pip install fastapi uvicorn

Calistirma:
    cd tua_sopranos1
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                       health check
    GET  /api/status             sistem ozeti (hafif)
    GET  /api/full               tam UI verisi (onbellekli)
    GET  /api/satellites         uydu listesi (yol verisi yok)
    GET  /api/satellite/{name}   tek uydu tam detayi
    GET  /api/threats/{name}     uydu tehdit listesi
    POST /api/refresh            onbellegi zorla yenile
"""

import sys
import os
import datetime
import threading
import time
from typing import Optional

# ── Path ayari ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, "Veri_analizi"))

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Kendi modulumuz ───────────────────────────────────────────────────────────
from ui_data_generator import generate_ui_data
from Veri_analizi.config import TURKISH_SATELLITES

# ═════════════════════════════════════════════════════════════════════════════
# UYGULAMA
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "TUA SOPRANOS API",
    description = "Turk uydu carpisma risk analizi ve yorunge tahmin servisi",
    version     = "1.0.0",
)

# CORS — React (localhost:3000) + herhangi bir origin'e izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ═════════════════════════════════════════════════════════════════════════════
# ONBELLEK
# ═════════════════════════════════════════════════════════════════════════════

_CACHE: dict = {
    "data":       None,
    "generated":  None,  # datetime
    "is_loading": False,
}
_CACHE_TTL_SECONDS = 600   # 10 dakika
_cache_lock = threading.Lock()


def _cache_is_fresh() -> bool:
    if _CACHE["data"] is None or _CACHE["generated"] is None:
        return False
    age = (datetime.datetime.utcnow() - _CACHE["generated"]).total_seconds()
    return age < _CACHE_TTL_SECONDS


def _refresh_cache(use_cache: bool = True) -> None:
    """Arka planda veri uretir ve onbellegi gunceller."""
    with _cache_lock:
        if _CACHE["is_loading"]:
            return
        _CACHE["is_loading"] = True

    try:
        data = generate_ui_data(use_cache=use_cache)
        with _cache_lock:
            _CACHE["data"]      = data
            _CACHE["generated"] = datetime.datetime.utcnow()
    except Exception as exc:
        with _cache_lock:
            _CACHE["data"] = {"error": str(exc), "satellites": []}
    finally:
        with _cache_lock:
            _CACHE["is_loading"] = False


def _get_cached_data(background_tasks: BackgroundTasks) -> dict:
    """
    Onbellekten veri dondurur.
    Onbellek bayatsa arka planda yeniler, simdilik mevcut veriyi dondurur.
    Hic veri yoksa senkron olarak uretir (ilk istek).
    """
    if _CACHE["data"] is None:
        # Ilk istek — senkron uret
        _refresh_cache(use_cache=True)
    elif not _cache_is_fresh():
        # Eski veri var, arka planda yenile
        background_tasks.add_task(_refresh_cache, use_cache=True)

    return _CACHE["data"] or {}


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Uygulama baslarken arka planda ilk veriyi yukle."""
    t = threading.Thread(target=_refresh_cache, args=(True,), daemon=True)
    t.start()


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    """API saglik kontrolu."""
    cache_age = None
    if _CACHE["generated"]:
        cache_age = round(
            (datetime.datetime.utcnow() - _CACHE["generated"]).total_seconds()
        )
    return {
        "service":    "TUA SOPRANOS API",
        "version":    "1.0.0",
        "status":     "online",
        "cache_age_s": cache_age,
        "cache_fresh": _cache_is_fresh(),
        "timestamp":  datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: SISTEM DURUMU (HAFIF)
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/status", tags=["Overview"])
def get_status(background_tasks: BackgroundTasks):
    """
    Sistem ozeti — buyuk yol verisi olmadan.
    Dashboard baslik bant genisligi duyarli istemciler icin.
    """
    data = _get_cached_data(background_tasks)
    if "error" in data:
        raise HTTPException(status_code=503, detail=data["error"])

    return {
        "generated_at":  data.get("generated_at"),
        "system_status": data.get("system_status", "UNKNOWN"),
        "models_status": data.get("models_status", {}),
        "summary":       data.get("summary", {}),
        "satellites": [
            {
                "name":         s.get("name"),
                "orbit_type":   s.get("orbit_type"),
                "status":       s.get("status"),
                "threat_level": s.get("threat_level", "GREEN"),
                "n_threats":    len(s.get("threats", [])),
                "can_maneuver": s.get("can_maneuver", False),
                "tle_confidence": s.get("tle_confidence"),
            }
            for s in data.get("satellites", [])
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: TAM VERİ
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/full", tags=["Data"])
def get_full(background_tasks: BackgroundTasks):
    """
    Tum UI verisi — orbit yollari, tehditler, manevra onerileri dahil.
    Ilk yuklemede 30-60 saniye surebilir (TLE + SGP4 + LSTM).
    Onbellekten sonra aninda doner.
    """
    data = _get_cached_data(background_tasks)
    if "error" in data and not data.get("satellites"):
        raise HTTPException(status_code=503, detail=data["error"])
    return data


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: UYDU LİSTESİ
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/satellites", tags=["Satellites"])
def get_satellites(background_tasks: BackgroundTasks):
    """
    Tum Turk uydularinin listesi.
    Orbit yol verisi (orbit_path) dahil, predicted_path haric — bant tasarrufu.
    """
    data = _get_cached_data(background_tasks)
    if "error" in data and not data.get("satellites"):
        raise HTTPException(status_code=503, detail=data["error"])

    result = []
    for s in data.get("satellites", []):
        entry = {k: v for k, v in s.items() if k != "predicted_path"}
        result.append(entry)

    return {"satellites": result, "count": len(result)}


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: TEK UYDU DETAYI
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/satellite/{sat_name}", tags=["Satellites"])
def get_satellite(sat_name: str, background_tasks: BackgroundTasks):
    """
    Tek uydu icin tam detay — orbit yolu, tahmin yolu, tehditler, manevra dahil.
    sat_name ornegi: Turksat%205A  (URL encode gerekebilir)
    """
    data = _get_cached_data(background_tasks)
    if "error" in data and not data.get("satellites"):
        raise HTTPException(status_code=503, detail=data["error"])

    # URL encoding normalize et
    name_query = sat_name.replace("%20", " ").strip()

    for sat in data.get("satellites", []):
        if sat.get("name", "").lower() == name_query.lower():
            return sat

    # Bilinen uydu ama henuz veri yok
    known = list(TURKISH_SATELLITES.keys())
    raise HTTPException(
        status_code=404,
        detail=f"'{name_query}' bulunamadi. Bilinen uydular: {known}",
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: TEHDIT LİSTESİ
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/threats/{sat_name}", tags=["Threats"])
def get_threats(
    sat_name:           str,
    background_tasks:   BackgroundTasks,
    min_level:          str = "WATCH",
):
    """
    Belirli bir uyduya ait tehdit listesi.

    Query params:
        min_level : Minimum seviye filtresi — GREEN|WATCH|YELLOW|RED (varsayilan: WATCH)
    """
    data = _get_cached_data(background_tasks)
    if "error" in data and not data.get("satellites"):
        raise HTTPException(status_code=503, detail=data["error"])

    name_query = sat_name.replace("%20", " ").strip()
    sat        = next(
        (s for s in data.get("satellites", [])
         if s.get("name", "").lower() == name_query.lower()),
        None,
    )
    if sat is None:
        raise HTTPException(status_code=404, detail=f"'{name_query}' bulunamadi")

    all_threats = sat.get("threats", [])

    # Seviye filtresi
    level_rank  = {"GREEN": 0, "WATCH": 1, "YELLOW": 2, "RED": 3}
    min_rank    = level_rank.get(min_level.upper(), 1)
    filtered    = [
        t for t in all_threats
        if level_rank.get(t["classification"]["label"], 0) >= min_rank
    ]

    return {
        "sat_name":     sat.get("name"),
        "orbit_type":   sat.get("orbit_type"),
        "threat_level": sat.get("threat_level"),
        "total":        len(all_threats),
        "filtered":     len(filtered),
        "min_level":    min_level.upper(),
        "threats":      filtered,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: CACHE YENILE
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/api/refresh", tags=["Admin"])
def refresh_cache(
    background_tasks: BackgroundTasks,
    force_api:        bool = False,
):
    """
    Onbellegi zorla yenile.
    force_api=true ise Space-Track API'den ceker (cache degil).
    """
    if _CACHE["is_loading"]:
        return {"status": "already_loading", "message": "Veri zaten yukleniyor"}

    background_tasks.add_task(_refresh_cache, not force_api)
    return {
        "status":    "refresh_started",
        "message":   "Veri arka planda yenileniyor",
        "force_api": force_api,
    }


# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINT: SISTEM BİLGİSİ
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/info", tags=["Overview"])
def get_info():
    """API ve sistem bilgisi."""
    from model.ml_model import load_model, load_lstm_model
    xgb        = load_model()
    lstm, scal = load_lstm_model()
    return {
        "project":    "TUA SOPRANOS",
        "api_version": "1.0.0",
        "models": {
            "xgboost": {
                "status":    "LOADED" if xgb else "MISSING",
                "accuracy":  xgb.get("accuracy") if xgb else None,
                "n_samples": xgb.get("n_samples") if xgb else None,
            },
            "lstm": {
                "status": "LOADED" if lstm else "MISSING",
            },
        },
        "satellites": {
            "total":   len(TURKISH_SATELLITES),
            "names":   list(TURKISH_SATELLITES.keys()),
        },
        "cache": {
            "ttl_seconds": _CACHE_TTL_SECONDS,
            "is_fresh":    _cache_is_fresh(),
            "generated":   _CACHE["generated"].strftime("%Y-%m-%dT%H:%M:%SZ")
                           if _CACHE["generated"] else None,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# DOĞRUDAN CALISTIRMA
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("TUA SOPRANOS API baslatiliyor...")
    print("Dokumantasyon: http://localhost:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
