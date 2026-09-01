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
import math
import datetime
import threading
import time
from typing import Optional

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Path ayari ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
_VERI_DIR = os.path.join(_BASE, "Veri_analizi")
if _VERI_DIR not in sys.path:
    sys.path.insert(0, _VERI_DIR)

# ── FastAPI ───────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
# ENDPOINT: MANEVRA SİMÜLASYONU  (Türksat 6A senaryo hesaplayıcı)
# ═════════════════════════════════════════════════════════════════════════════

# ── Fizik sabitleri ───────────────────────────────────────────────────────────
_G0              = 9.80665        # standart yer çekimi ivmesi  [m/s²]
_ISP_SEC         = 220.0          # hidrazin mono-prop Isp       [s]
_THRUST_N        = 10.0           # tipik GEO durum itki gücü    [N]
_HYDRAZINE_USD   = 15_500         # uçuş onaylı hidrazin maliyeti [$/kg]

# Manevra tipi başına kilometre kazancı ölçekleme katsayısı:
#   improvement_km ≈ delta_v_ms × tca_hours × scale
# Prograde / Retrograde en verimli (boylamsal ayrışma).
# Radyal daha az etkili. Çift-impuls kısmi iptal nedeniyle orta.
_TYPE_SCALE = {
    "Prograde":     13.5,
    "Retrograde":   13.5,
    "Radial Out":    8.0,
    "Radial In":     7.5,
    "Bi-Impulsive":  5.5,
}

# Manevra tipi başına görev ömrü darbesi çarpanı:
#   impact_days = fuel_kg × 15 × multiplier
_TYPE_LIFE_MULT = {
    "Prograde":    1.00,
    "Retrograde":  1.00,
    "Radial Out":  1.00,
    "Radial In":   1.00,
    "Bi-Impulsive": 0.35,   # dönüş yanımı yakıtı geri kazanır
}


class ManeuverRequest(BaseModel):
    """POST /api/simulate_maneuver için istek modeli."""
    sat_name:                str            # ör. "Turksat 6A"
    scenario_id:             str            # ör. "micro_nudge"
    maneuver_type:           str            # "Prograde" | "Retrograde" | "Radial Out" | "Bi-Impulsive"
    delta_v_ms:              float          # Δv büyüklüğü  [m/s]
    tca_hours:               float = 18.0  # TCA'ya kalan süre  [saat]
    original_miss_distance_km: float = 0.3 # mevcut minimum mesafe  [km]
    initial_pc:              float = 1.83e-4  # mevcut çarpışma olasılığı


def _simulate_maneuver_physics(
    sat_name:                str,
    maneuver_type:           str,
    delta_v_ms:              float,
    tca_hours:               float,
    original_miss_km:        float,
    initial_pc:              float,
    cached_data:             dict,
) -> dict:
    """
    Tsiolkovsky roketi denklemi ve basitleştirilmiş yörünge yaklaşımlarını
    kullanarak manevra metriklerini hesaplar.

    Hesaplanan değerler:
        burn_duration_sec   — Tsiolkovsky'den kütle akışı ile
        fuel_mass_kg        — Tsiolkovsky'den
        fuel_efficiency_pct — Δv büyüklüğüne bağlı ampirik
        new_miss_distance   — lineer zamanla ayrışma yaklaşımı
        pc_after            — üstel ölçekli azalma modeli
        status_after        — Pc eşiklerine göre
        mission_life_impact — yakıt oranı × görev ömrü
        fuel_remaining_pct  — toplam kapasiteden
        fuel_cost_usd       — kg başına hidrazin fiyatı
        recommended         — en verimli (kg başına max km/kg)
    """

    # ── 1. Uydu parametrelerini önbellekten al ─────────────────────────────
    sat_info   = {}
    name_lower = sat_name.lower()
    for s in cached_data.get("satellites", []):
        if s.get("name", "").lower() == name_lower:
            sat_info = s
            break

    # Kütleye dayalı geri dönüşler (varsayılanlar Türksat 6A için makul)
    dry_mass_kg     = sat_info.get("mass_kg", 3500.0)
    current_fuel_kg = sat_info.get("fuel_kg", 890.0)

    # Toplam kapasite: yakıt oranı ~%35 (tipik GEO başlangıç kütlesi)
    fuel_capacity_kg = dry_mass_kg * 0.35
    fuel_capacity_kg = max(fuel_capacity_kg, current_fuel_kg + 10.0)

    # Toplam ıslak kütle (yakıt dahil)
    wet_mass_kg = dry_mass_kg + current_fuel_kg

    # ── 2. Tsiolkovsky roketi denklemi — yakıt kütlesi ────────────────────
    # m_prop = m_wet × (1 − exp(−|Δv| / (Isp × g0)))
    effective_exhaust_velocity = _ISP_SEC * _G0          # m/s
    mass_ratio  = 1.0 - math.exp(-delta_v_ms / effective_exhaust_velocity)
    fuel_mass_kg = wet_mass_kg * mass_ratio

    # ── 3. Yanma süresi — kütle akışından ─────────────────────────────────
    # ṁ = F / (Isp × g0)    [kg/s]
    mass_flow_rate   = _THRUST_N / effective_exhaust_velocity   # kg/s
    burn_duration_sec = fuel_mass_kg / mass_flow_rate            # s

    # ── 4. Yanma verimliliği — ampirik ────────────────────────────────────
    # Büyük Δv'ler daha uzun, daha az verimli yanma gerektirir.
    # 0 m/s → %98, 1 m/s → ~%78, minimum %72
    fuel_efficiency_pct = max(72.0, round(98.0 - delta_v_ms * 20.0, 1))

    # ── 5. Yeni ıska mesafesi — lineer ayrışma yaklaşımı ──────────────────
    # improvement_km ≈ Δv [m/s] × TCA [saat] × tip_katsayısı
    scale = _TYPE_SCALE.get(maneuver_type, 10.0)
    improvement_km       = delta_v_ms * tca_hours * scale
    new_miss_distance_km = original_miss_km + improvement_km

    # ── 6. Manevra sonrası çarpışma olasılığı ─────────────────────────────
    # Üstel ölçek: Pc_sonra = Pc_başlangıç × exp(−d_yeni / σ)
    # σ ≈ 2 km (GEO için tipik birleşik kovaryans sigma değeri)
    sigma_km = 2.0
    pc_after_float = initial_pc * math.exp(-new_miss_distance_km / sigma_km)
    pc_after_float = max(pc_after_float, 1e-14)   # sayısal alt sınır

    # Bilimsel gösterim dizisi
    pc_after_str = f"{pc_after_float:.2e}"
    if pc_after_float < 1e-12:
        pc_after_str = f"< 1e-12"

    # ── 7. Manevra sonrası durum eşikleri ─────────────────────────────────
    if pc_after_float < 1e-6:
        status_after = "GREEN"
    elif pc_after_float < 1e-4:
        status_after = "YELLOW"
    else:
        status_after = "RED"

    # ── 8. Görev ömrü etkisi ──────────────────────────────────────────────
    # impact_days = yakıt_kg × 15 × tip_çarpanı
    life_mult          = _TYPE_LIFE_MULT.get(maneuver_type, 1.0)
    mission_life_impact_days = max(1, round(fuel_mass_kg * 15.0 * life_mult))

    # ── 9. Kalan yakıt yüzdesi — toplam kapasiteden ───────────────────────
    fuel_after_burn    = max(0.0, current_fuel_kg - fuel_mass_kg)
    fuel_remaining_pct = round((fuel_after_burn / fuel_capacity_kg) * 100.0, 1)

    # ── 10. USD maliyet ───────────────────────────────────────────────────
    fuel_cost_usd = round(fuel_mass_kg * _HYDRAZINE_USD)

    # ── 11. Önerilen senaryo mu? ──────────────────────────────────────────
    # En iyi senaryo = kilogram başına en yüksek ıska kazancı VE TCA > 12 saat
    km_per_kg   = improvement_km / max(fuel_mass_kg, 1e-9)
    recommended = (delta_v_ms <= 0.08) and (tca_hours >= 12.0) and (km_per_kg > 50.0)

    return {
        "sat_name":               sat_name,
        "scenario_id":            None,       # çağıran tarafından doldurulur
        "type":                   maneuver_type,
        "delta_v_ms":             round(delta_v_ms, 4),
        "burn_duration_sec":      round(burn_duration_sec, 1),
        "fuel_mass_kg":           round(fuel_mass_kg, 4),
        "isp_sec":                _ISP_SEC,
        "fuel_efficiency_pct":    fuel_efficiency_pct,
        "new_miss_distance_km":   round(new_miss_distance_km, 2),
        "miss_improvement_km":    round(improvement_km, 2),
        "pc_after":               pc_after_str,
        "pc_after_float":         pc_after_float,
        "status_after":           status_after,
        "mission_life_impact_days": mission_life_impact_days,
        "fuel_remaining_pct":     fuel_remaining_pct,
        "fuel_cost_usd":          fuel_cost_usd,
        "recommended":            recommended,
        # meta — hata ayıklama / şeffaflık için
        "_wet_mass_kg":           round(wet_mass_kg, 1),
        "_fuel_capacity_kg":      round(fuel_capacity_kg, 1),
        "_current_fuel_kg":       round(current_fuel_kg, 1),
        "_km_per_kg":             round(km_per_kg, 2),
    }


@app.post("/api/simulate_maneuver", tags=["Maneuver"])
def simulate_maneuver(req: ManeuverRequest, background_tasks: BackgroundTasks):
    """
    Seçili uydu ve manevra senaryosu için fizik tabanlı manevra simülasyonu.

    Girdi:
        sat_name, scenario_id, maneuver_type, delta_v_ms,
        tca_hours, original_miss_distance_km, initial_pc

    Çıktı:
        Tsiolkovsky yakıt kütlesi, yanma süresi, ıska kazancı,
        Pc sonrası, durum etiketi, USD maliyeti, yakıt çubuğu %
    """
    # Uydu kütlesini ve yakıt miktarını çözmek için önbelleklenmiş veriyi kullan
    cached = _get_cached_data(background_tasks)

    result = _simulate_maneuver_physics(
        sat_name              = req.sat_name,
        maneuver_type         = req.maneuver_type,
        delta_v_ms            = req.delta_v_ms,
        tca_hours             = req.tca_hours,
        original_miss_km      = req.original_miss_distance_km,
        initial_pc            = req.initial_pc,
        cached_data           = cached,
    )
    result["scenario_id"] = req.scenario_id
    return result


class CDMRequest(BaseModel):
    primary_name:        str
    secondary_name:      str
    miss_distance_km:    float = 0.30
    pc:                  float = 1.83e-4
    tca_hours:           float = 18.0
    primary_norad_id:    Optional[str] = None
    secondary_norad_id:  Optional[str] = "99999"


@app.post("/api/cdm", tags=["Compliance"])
def get_cdm_xml(req: CDMRequest):
    """
    CCSDS 508.0-B-1 standardında resmi Conjunction Data Message (XML) üretir.
    """
    from model.cara_engine import generate_cdm
    tca_dt = datetime.datetime.utcnow() + datetime.timedelta(hours=req.tca_hours)
    assessment = {
        "pc":                req.pc,
        "pc_scientific":     f"{req.pc:.2e}",
        "miss_distance_km":  req.miss_distance_km,
        "miss_distance_m":   req.miss_distance_km * 1000.0,
        "time_to_tca_hours": req.tca_hours,
        "tca":               tca_dt.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "primary_name":      req.primary_name,
        "secondary_name":    req.secondary_name,
        "status":            "RED" if req.pc > 1e-4 else "YELLOW" if req.pc > 1e-5 else "GREEN",
    }
    norad_1 = req.primary_norad_id or str(TURKISH_SATELLITES.get(req.primary_name, {}).get("norad_id", "60233"))
    xml_str = generate_cdm(
        primary_name       = req.primary_name,
        secondary_name     = req.secondary_name,
        assessment         = assessment,
        primary_norad_id   = norad_1,
        secondary_norad_id = req.secondary_norad_id or "99999",
        originator         = "TUA_SOPRANOS_C2",
    )
    return {"xml": xml_str, "filename": f"CDM_{req.primary_name.replace(' ', '_')}.xml"}


# ═════════════════════════════════════════════════════════════════════════════
# DOĞRUDAN CALISTIRMA
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print("TUA SOPRANOS API baslatiliyor...")
    print("Dokumantasyon: http://localhost:8000/docs")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
