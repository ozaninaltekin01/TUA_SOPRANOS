# ============================================
# YÖRÜNGE KALKANI — VERİ ÇEKME MOTORU
# ============================================

import requests
import json
import os
import datetime
import numpy as np
from sgp4.api import Satrec, jday
from config import (
    SPACETRACK_USER, SPACETRACK_PASS,
    TURKISH_SATELLITES, DEBRIS_FILTERS,
    COVARIANCE_PARAMS, EARTH_RADIUS_KM
)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================
# 1. SPACE-TRACK BAĞLANTISI
# ============================================

class SpaceTrackClient:
    BASE_URL = "https://www.space-track.org"
    LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
    QUERY_URL = f"{BASE_URL}/basicspacedata/query"

    def __init__(self):
        self.session = requests.Session()
        self._logged_in = False

    def login(self):
        resp = self.session.post(self.LOGIN_URL, data={
            "identity": SPACETRACK_USER,
            "password": SPACETRACK_PASS
        })
        self._logged_in = resp.status_code == 200
        return self._logged_in

    def query(self, endpoint):
        if not self._logged_in:
            self.login()
        resp = self.session.get(f"{self.QUERY_URL}/{endpoint}")
        if resp.status_code == 200 and resp.text.strip():
            try:
                return resp.json()
            except:
                return []
        return []


# ============================================
# 2. TLE VERİSİ ÇEKME
# ============================================

def fetch_satellite_tle(client, norad_id):
    """Tek bir uydu için en güncel TLE çeker"""
    data = client.query(
        f"class/gp/NORAD_CAT_ID/{norad_id}"
        f"/orderby/EPOCH%20desc/limit/1/format/json"
    )
    if data and len(data) > 0:
        return data[0]
    return None


def fetch_all_turkish_tle(client):
    """Tüm Türk uydularının TLE verilerini çeker"""
    results = {}
    for name, info in TURKISH_SATELLITES.items():
        gp = fetch_satellite_tle(client, info["norad_id"])
        if gp and "TLE_LINE1" in gp:
            results[name] = {
                "line1": gp["TLE_LINE1"],
                "line2": gp["TLE_LINE2"],
                "epoch": gp.get("EPOCH", ""),
                "object_name": gp.get("OBJECT_NAME", name),
                "rcs_size": gp.get("RCS_SIZE", "MEDIUM"),
                "norad_id": info["norad_id"],
                "orbit": info["orbit"],
                "type": info["type"],
                "status": info["status"],
                "mass_kg": info["mass_kg"],
            }
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} — veri alınamadı")
    return results


# ============================================
# 3. ÇÖP TARAMA
# ============================================

def fetch_debris(client, orbit_type="GEO"):
    """GEO veya LEO çöplerini çeker"""
    f = DEBRIS_FILTERS[orbit_type]
    data = client.query(
        f"class/gp"
        f"/MEAN_MOTION/{f['mean_motion_min']}--{f['mean_motion_max']}"
        f"/EPOCH/%3Enow-1"
        f"/orderby/NORAD_CAT_ID"
        f"/limit/{f['limit']}"
        f"/format/json"
    )
    results = []
    for obj in data:
        if "TLE_LINE1" not in obj or "TLE_LINE2" not in obj:
            continue
        results.append({
            "line1": obj["TLE_LINE1"],
            "line2": obj["TLE_LINE2"],
            "epoch": obj.get("EPOCH", ""),
            "object_name": obj.get("OBJECT_NAME", "UNKNOWN"),
            "norad_id": obj.get("NORAD_CAT_ID", ""),
            "object_type": obj.get("OBJECT_TYPE", "UNKNOWN"),
            "country": obj.get("COUNTRY_CODE", "UNK"),
            "rcs_size": obj.get("RCS_SIZE", "MEDIUM"),
            "launch_year": obj.get("LAUNCH_DATE", "")[:4] if obj.get("LAUNCH_DATE") else "",
        })
    print(f"  {orbit_type} çöp sayısı: {len(results)}")
    return results


# ============================================
# 4. POZİSYON HESAPLAMA (SGP4)
# ============================================

def calculate_position(line1, line2, dt=None):
    """TLE'den pozisyon ve hız hesaplar"""
    if dt is None:
        dt = datetime.datetime.utcnow()
    try:
        sat = Satrec.twoline2rv(line1, line2)
        jd, fr = jday(dt.year, dt.month, dt.day,
                       dt.hour, dt.minute, dt.second)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0 or any(np.isnan(r)) or any(np.isnan(v)):
            return None
        altitude = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - EARTH_RADIUS_KM
        speed = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        return {
            "pos": list(r),
            "vel": list(v),
            "altitude_km": round(altitude, 2),
            "speed_kms": round(speed, 4),
        }
    except Exception:
        return None


def calculate_positions_batch(objects_dict):
    """Birden fazla nesne için toplu pozisyon hesabı"""
    now = datetime.datetime.utcnow()
    results = {}
    for name, obj in objects_dict.items():
        pos = calculate_position(obj["line1"], obj["line2"], now)
        if pos:
            pos["name"] = name
            pos["norad_id"] = obj.get("norad_id", "")
            pos["object_type"] = obj.get("object_type", obj.get("type", ""))
            pos["country"] = obj.get("country", "TUR")
            pos["rcs_size"] = obj.get("rcs_size", "MEDIUM")
            pos["epoch"] = obj.get("epoch", "")
            results[name] = pos
    return results


def calculate_positions_list(objects_list):
    """Liste formatındaki nesneler için toplu pozisyon hesabı"""
    now = datetime.datetime.utcnow()
    results = []
    for obj in objects_list:
        pos = calculate_position(obj["line1"], obj["line2"], now)
        if pos:
            pos["name"] = obj.get("object_name", "UNKNOWN")
            pos["norad_id"] = obj.get("norad_id", "")
            pos["object_type"] = obj.get("object_type", "UNKNOWN")
            pos["country"] = obj.get("country", "UNK")
            pos["rcs_size"] = obj.get("rcs_size", "MEDIUM")
            pos["epoch"] = obj.get("epoch", "")
            results.append(pos)
    return results


# ============================================
# 5. SENTETİK KOVARYANS MATRİSİ (CARA UYUMLU)
# ============================================

def generate_covariance(epoch_str, orbit_type="LEO"):
    """TLE yaşından sentetik kovaryans matrisi üretir"""
    try:
        epoch_dt = datetime.datetime.strptime(epoch_str[:19], "%Y-%m-%dT%H:%M:%S")
        age_hours = (datetime.datetime.utcnow() - epoch_dt).total_seconds() / 3600
    except:
        age_hours = 24.0  # varsayılan

    sigma_rate = COVARIANCE_PARAMS.get(orbit_type, COVARIANCE_PARAMS["LEO"])["pos_sigma_per_hour"]
    sigma_pos = max(sigma_rate * age_hours, 0.05)  # minimum 50 metre

    # 3x3 diagonal kovaryans (km²)
    cov = np.diag([sigma_pos**2, sigma_pos**2, sigma_pos**2])
    return cov, sigma_pos, age_hours


# ============================================
# 6. HARD BODY RADIUS (HBR)
# ============================================

def estimate_hbr(rcs_size="MEDIUM", mass_kg=None):
    """RCS boyutundan veya kütleden HBR tahmin eder (metre)"""
    rcs_map = {
        "SMALL":  0.5,
        "MEDIUM": 1.5,
        "LARGE":  3.0,
    }
    if mass_kg and mass_kg > 0:
        # kütle bazlı yaklaşık boyut (küresel varsayım, yoğunluk ~100 kg/m³)
        radius_m = (3 * mass_kg / (4 * 3.14159 * 100)) ** (1/3)
        return max(radius_m, 0.1)
    return rcs_map.get(rcs_size, 1.5)


# ============================================
# 7. TLE GÜVEN SKORU (#18)
# ============================================

def tle_confidence_score(epoch_str, orbit_type="LEO"):
    """TLE yaşına göre güven skoru hesaplar (0-100)"""
    try:
        epoch_dt = datetime.datetime.strptime(epoch_str[:19], "%Y-%m-%dT%H:%M:%S")
        age_hours = (datetime.datetime.utcnow() - epoch_dt).total_seconds() / 3600
    except:
        return 10.0, 999

    # LEO'da TLE hızlı bozulur, GEO'da yavaş
    if orbit_type == "LEO":
        half_life = 6.0   # 6 saatte %50 güven kaybı
    else:
        half_life = 48.0  # 48 saatte %50 güven kaybı

    score = 100 * (0.5 ** (age_hours / half_life))
    return round(max(score, 1.0), 1), round(age_hours, 1)


# ============================================
# 8. MESAFE HESAPLAMA
# ============================================

def calculate_distance(pos1, pos2):
    """İki pozisyon arası mesafe (km)"""
    r1 = pos1["pos"]
    r2 = pos2["pos"]
    return np.sqrt(
        (r1[0]-r2[0])**2 +
        (r1[1]-r2[1])**2 +
        (r1[2]-r2[2])**2
    )


def find_closest_threats(target_pos, debris_positions, n=10, exclude_turkish=True):
    """Hedef uyduya en yakın n tehdidi bulur"""
    threats = []
    for deb in debris_positions:
        if exclude_turkish and deb.get("country") == "TUR":
            continue
        dist = calculate_distance(target_pos, deb)
        threats.append({
            "name": deb["name"],
            "norad_id": deb["norad_id"],
            "object_type": deb["object_type"],
            "country": deb["country"],
            "distance_km": round(dist, 2),
            "pos": deb["pos"],
            "vel": deb["vel"],
            "rcs_size": deb.get("rcs_size", "MEDIUM"),
            "epoch": deb.get("epoch", ""),
        })
    threats.sort(key=lambda x: x["distance_km"])
    return threats[:n]


# ============================================
# 9. CACHE SİSTEMİ (demo güvenliği)
# ============================================

def save_cache(data, filename):
    """Veriyi JSON olarak cache'le"""
    path = os.path.join(CACHE_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, default=str, indent=2)
    print(f"  💾 Cache kaydedildi: {filename}")


def load_cache(filename):
    """Cache'ten veri yükle"""
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def refresh_cache(client=None):
    """Tüm verileri tazele ve cache'le"""
    if client is None:
        client = SpaceTrackClient()
        if not client.login():
            print("❌ Space-Track login başarısız!")
            return False

    print("\n📡 Türk uyduları çekiliyor...")
    turkish = fetch_all_turkish_tle(client)
    save_cache(turkish, "turkish_tle.json")

    print("\n📡 GEO çöpleri çekiliyor...")
    geo_debris = fetch_debris(client, "GEO")
    save_cache(geo_debris, "geo_debris.json")

    print("\n📡 LEO çöpleri çekiliyor...")
    leo_debris = fetch_debris(client, "LEO")
    save_cache(leo_debris, "leo_debris.json")

    print("\n✅ Tüm cache güncel!")
    return True


# ============================================
# 10. ANA VERİ YÜKLEYİCİ
# ============================================

def load_all_data(use_cache=False):
    """Tüm veriyi yükler — cache varsa oradan, yoksa API'den"""

    if use_cache:
        turkish = load_cache("turkish_tle.json")
        geo_debris = load_cache("geo_debris.json")
        leo_debris = load_cache("leo_debris.json")
        if turkish and geo_debris and leo_debris:
            print("📂 Cache'ten yüklendi")
            return turkish, geo_debris, leo_debris
        print("⚠️ Cache eksik, API'ye geçiliyor...")

    client = SpaceTrackClient()
    if not client.login():
        print("❌ Login başarısız! Cache deneniyor...")
        return load_all_data(use_cache=True)

    turkish = fetch_all_turkish_tle(client)
    geo_debris = fetch_debris(client, "GEO")
    leo_debris = fetch_debris(client, "LEO")

    # Otomatik cache'le
    save_cache(turkish, "turkish_tle.json")
    save_cache(geo_debris, "geo_debris.json")
    save_cache(leo_debris, "leo_debris.json")

    return turkish, geo_debris, leo_debris


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    print("=" * 55)
    print("🛰️  YÖRÜNGE KALKANI — VERİ MOTORU TESTİ")
    print("=" * 55)

    # Veri yükle
    turkish, geo_debris, leo_debris = load_all_data()

    # Pozisyon hesapla
    print(f"\n📍 Türk uyduları pozisyon hesabı...")
    tr_positions = calculate_positions_batch(turkish)
    for name, pos in tr_positions.items():
        conf, age = tle_confidence_score(
            turkish[name]["epoch"],
            turkish[name]["orbit"]
        )
        print(f"  {name:<15} alt={pos['altitude_km']:>10.1f} km  "
              f"hız={pos['speed_kms']:.3f} km/s  "
              f"TLE güven={conf}% ({age:.0f}h)")

    # GEO çöp pozisyonları
    print(f"\n🗑️  GEO çöp pozisyon hesabı...")
    geo_positions = calculate_positions_list(geo_debris)
    print(f"  Hesaplanan: {len(geo_positions)} nesne")

    # LEO çöp pozisyonları
    print(f"\n🗑️  LEO çöp pozisyon hesabı...")
    leo_positions = calculate_positions_list(leo_debris)
    print(f"  Hesaplanan: {len(leo_positions)} nesne")

    # En yakın tehditler
    print(f"\n🎯 Tehdit analizi...")
    for name, pos in tr_positions.items():
        orbit = turkish[name]["orbit"]
        debris_pool = geo_positions if orbit == "GEO" else leo_positions
        threats = find_closest_threats(pos, debris_pool, n=3)
        print(f"\n  {name}:")
        for i, t in enumerate(threats, 1):
            print(f"    {i}. {t['name']:<25} {t['distance_km']:>10.1f} km  "
                  f"[{t['object_type']}] ({t['country']})")

    print(f"\n{'=' * 55}")
    print("✅ Veri motoru testi tamamlandı!")