# ============================================
# YÖRÜNGE KALKANI — TEHDİT ANALİZ MOTORLARI
# ============================================

import numpy as np
import datetime
from config import TURKISH_SATELLITES, EARTH_RADIUS_KM


# ============================================
# FEATURE #6 — GEO KOMŞULUK ANALİZİ
# ============================================

def calculate_subsatellite_longitude(pos):
    """
    ECI pozisyonundan sub-satellite longitude hesaplar.
    GEO uydularının boylam slotunu bulmak için kullanılır.
    """
    x, y, z = pos[0], pos[1], pos[2]

    # ECI → ECEF dönüşümü (dünya dönüşü hesaba katılır)
    now = datetime.datetime.utcnow()
    # Greenwich Sidereal Time (basitleştirilmiş)
    j2000 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    days_since_j2000 = (now - j2000).total_seconds() / 86400.0
    gmst_deg = (280.46061837 + 360.98564736629 * days_since_j2000) % 360

    # ECI → ECEF rotasyonu
    gmst_rad = np.radians(gmst_deg)
    x_ecef = x * np.cos(gmst_rad) + y * np.sin(gmst_rad)
    y_ecef = -x * np.sin(gmst_rad) + y * np.cos(gmst_rad)

    # Longitude
    lon_deg = np.degrees(np.arctan2(y_ecef, x_ecef))
    return round(lon_deg, 2)


def geo_neighborhood_analysis(turkish_positions, turkish_tle_data,
                                geo_debris_positions, geo_debris_raw,
                                longitude_range_deg=5.0):
    """
    Her Türksat GEO uydusu için komşuluk analizi yapar.
    
    ±longitude_range_deg içindeki tüm nesneleri bulur ve
    aktif/ölü/çöp olarak sınıflandırır.
    
    Çıktı: her Türksat uydusu için komşu listesi + ölü komşu skoru
    """
    results = {}

    # Önce tüm GEO nesnelerinin boylamlarını hesapla
    debris_with_lon = []
    for deb in geo_debris_positions:
        lon = calculate_subsatellite_longitude(deb["pos"])
        deb_copy = dict(deb)
        deb_copy["longitude"] = lon
        debris_with_lon.append(deb_copy)

    # Her GEO Türksat uydusu için analiz
    for name, pos in turkish_positions.items():
        sat_info = TURKISH_SATELLITES.get(name, {})
        if sat_info.get("orbit") != "GEO":
            continue

        sat_lon = calculate_subsatellite_longitude(pos["pos"])
        altitude = pos["altitude_km"]

        # Komşuları bul (±range derece)
        neighbors = []
        for deb in debris_with_lon:
            lon_diff = abs(deb["longitude"] - sat_lon)
            # 360° wrap-around kontrolü
            if lon_diff > 180:
                lon_diff = 360 - lon_diff

            if lon_diff <= longitude_range_deg:
                # Nesne tipini belirle
                obj_type = deb.get("object_type", "UNKNOWN")
                is_debris = obj_type in ["DEBRIS", "ROCKET BODY"]
                is_dead_sat = obj_type == "PAYLOAD" and _is_likely_dead(deb)

                threat_level = "LOW"
                if is_debris:
                    threat_level = "MEDIUM"
                if is_dead_sat:
                    threat_level = "HIGH"  # ölü uydu kontrolsüz sürüklenir

                neighbors.append({
                    "name": deb["name"],
                    "norad_id": deb.get("norad_id", ""),
                    "longitude": deb["longitude"],
                    "lon_diff_deg": round(lon_diff, 3),
                    "object_type": obj_type,
                    "country": deb.get("country", "UNK"),
                    "is_debris": is_debris,
                    "is_dead_satellite": is_dead_sat,
                    "threat_level": threat_level,
                    "distance_km": _approx_geo_distance(lon_diff),
                })

        # Sırala: boylam farkına göre
        neighbors.sort(key=lambda x: x["lon_diff_deg"])

        # Ölü komşu skoru hesapla
        dead_count = sum(1 for n in neighbors if n["is_dead_satellite"])
        debris_count = sum(1 for n in neighbors if n["is_debris"])
        total_count = len(neighbors)

        # Risk skoru: ölü uydu × 3 + çöp × 1 (ölü uydu daha tehlikeli çünkü büyük ve kontrolsüz)
        dead_neighbor_score = dead_count * 3 + debris_count * 1

        # Tehlike seviyesi
        if dead_neighbor_score >= 10:
            zone_status = "CRITICAL"
        elif dead_neighbor_score >= 5:
            zone_status = "ELEVATED"
        elif dead_neighbor_score >= 1:
            zone_status = "MODERATE"
        else:
            zone_status = "CLEAR"

        results[name] = {
            "satellite_longitude": sat_lon,
            "satellite_altitude_km": round(altitude, 1),
            "search_range_deg": longitude_range_deg,
            "total_neighbors": total_count,
            "dead_satellites": dead_count,
            "debris_objects": debris_count,
            "active_satellites": total_count - dead_count - debris_count,
            "dead_neighbor_score": dead_neighbor_score,
            "zone_status": zone_status,
            "neighbors": neighbors,

            # Ülke bazlı dağılım
            "country_breakdown": _country_breakdown(neighbors),
        }

    return results


def _is_likely_dead(obj):
    """
    Bir payload'ın ölü olup olmadığını tahmin eder.
    Basit heuristik: fırlatma yılı + beklenen ömür
    """
    launch_str = obj.get("launch_year", "")
    if not launch_str:
        return False
    try:
        launch_year = int(launch_str[:4]) if len(launch_str) >= 4 else 0
    except:
        return False
    
    current_year = datetime.datetime.utcnow().year
    age = current_year - launch_year

    # GEO uydusu ortalama ömrü 15 yıl
    # 20 yıldan eski GEO payload → muhtemelen ölü
    return age > 20


def _approx_geo_distance(lon_diff_deg):
    """GEO yüksekliğinde boylam farkından yaklaşık mesafe (km)"""
    geo_radius = EARTH_RADIUS_KM + 35786.0
    arc_km = geo_radius * np.radians(lon_diff_deg)
    return round(arc_km, 1)


def _country_breakdown(neighbors):
    """Komşuları ülkeye göre grupla"""
    countries = {}
    for n in neighbors:
        c = n["country"]
        if c not in countries:
            countries[c] = {"total": 0, "debris": 0, "dead_sat": 0}
        countries[c]["total"] += 1
        if n["is_debris"]:
            countries[c]["debris"] += 1
        if n["is_dead_satellite"]:
            countries[c]["dead_sat"] += 1
    return countries


def print_neighborhood_report(results):
    """Komşuluk analiz raporunu yazdırır"""
    for name, data in results.items():
        status_icon = {
            "CRITICAL": "🔴", "ELEVATED": "🟠",
            "MODERATE": "🟡", "CLEAR": "🟢"
        }.get(data["zone_status"], "⚪")

        print(f"\n{'='*60}")
        print(f"📍 {name} — Boylam: {data['satellite_longitude']}°E")
        print(f"   Yükseklik: {data['satellite_altitude_km']} km")
        print(f"   Arama alanı: ±{data['search_range_deg']}°")
        print(f"-"*60)
        print(f"   {status_icon} Bölge durumu: {data['zone_status']}")
        print(f"   Ölü komşu skoru: {data['dead_neighbor_score']}")
        print(f"   Toplam komşu: {data['total_neighbors']}")
        print(f"     Ölü uydu    : {data['dead_satellites']}")
        print(f"     Çöp/enkaz   : {data['debris_objects']}")
        print(f"     Aktif uydu  : {data['active_satellites']}")

        # Ülke dağılımı
        if data["country_breakdown"]:
            print(f"\n   Ülke dağılımı:")
            for country, counts in sorted(data["country_breakdown"].items(),
                                           key=lambda x: x[1]["total"], reverse=True):
                dead_str = f" ({counts['dead_sat']} ölü)" if counts["dead_sat"] > 0 else ""
                print(f"     {country:<6} {counts['total']} nesne{dead_str}")

        # En yakın 5 komşu
        print(f"\n   En yakın 5 komşu:")
        for n in data["neighbors"][:5]:
            threat_icon = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}.get(n["threat_level"], "⚪")
            dead_tag = " [ÖLÜ]" if n["is_dead_satellite"] else ""
            print(f"     {threat_icon} {n['name']:<28} "
                  f"Δlon={n['lon_diff_deg']:>6.2f}° "
                  f"~{n['distance_km']:>8.1f} km "
                  f"[{n['object_type']}] ({n['country']}){dead_tag}")


# ============================================
# TEST
# ============================================
# ============================================
# FEATURE #19 — PARÇALANMA ERKEN UYARISI
# ============================================

def fetch_tle_history(client, norad_id, days=30):
    """
    Bir nesnenin son N günlük TLE geçmişini çeker.
    BSTAR ve orbital eleman trendlerini analiz etmek için.
    """
    data = client.query(
        f"class/gp_history/NORAD_CAT_ID/{norad_id}"
        f"/orderby/EPOCH%20desc/limit/30/format/json"
    )
    if not data:
        return []

    history = []
    for gp in data:
        try:
            history.append({
                "epoch": gp.get("EPOCH", ""),
                "bstar": float(gp.get("BSTAR", 0)),
                "eccentricity": float(gp.get("ECCENTRICITY", 0)),
                "inclination": float(gp.get("INCLINATION", 0)),
                "mean_motion": float(gp.get("MEAN_MOTION", 0)),
                "ra_of_asc_node": float(gp.get("RA_OF_ASC_NODE", 0)),
                "arg_of_pericenter": float(gp.get("ARG_OF_PERICENTER", 0)),
                "norad_id": norad_id,
            })
        except:
            continue

    return sorted(history, key=lambda x: x["epoch"])


def analyze_breakup_risk(tle_history, object_name="UNKNOWN"):
    """
    TLE geçmişinden parçalanma riski analiz eder.
    
    Parçalanma öncesi belirtiler:
    1. BSTAR'da ani artış (atmosfer etkileşimi veya gaz kaçağı)
    2. Eccentricity'de anormal değişim (yörünge bozulması)
    3. Mean motion'da atlama (yükseklik değişimi)
    
    Çıktı: risk skoru (0-100) ve anomali detayları
    """
    if len(tle_history) < 5:
        return {
            "object_name": object_name,
            "risk_score": 0,
            "status": "INSUFFICIENT_DATA",
            "anomalies": [],
            "data_points": len(tle_history),
        }

    anomalies = []
    risk_score = 0

    # --- BSTAR trend analizi ---
    bstars = [h["bstar"] for h in tle_history]
    bstar_mean = np.mean(bstars)
    bstar_std = np.std(bstars) if len(bstars) > 1 else 0

    if bstar_std > 0 and bstar_mean != 0:
        # Son 5 değerin ortalaması vs toplam ortalama
        recent_bstar = np.mean(bstars[-5:])
        if bstar_std > 0:
            bstar_zscore = abs(recent_bstar - bstar_mean) / bstar_std
        else:
            bstar_zscore = 0

        if bstar_zscore > 3.0:
            anomalies.append({
                "type": "BSTAR_SPIKE",
                "severity": "HIGH",
                "detail": f"BSTAR z-score: {bstar_zscore:.1f} (>{3.0} eşik)",
                "value": recent_bstar,
                "baseline": bstar_mean,
            })
            risk_score += 35
        elif bstar_zscore > 2.0:
            anomalies.append({
                "type": "BSTAR_ELEVATED",
                "severity": "MEDIUM",
                "detail": f"BSTAR z-score: {bstar_zscore:.1f} (>{2.0} eşik)",
                "value": recent_bstar,
                "baseline": bstar_mean,
            })
            risk_score += 20

    # --- Eccentricity trend analizi ---
    eccs = [h["eccentricity"] for h in tle_history]
    ecc_mean = np.mean(eccs)
    ecc_std = np.std(eccs) if len(eccs) > 1 else 0

    if ecc_std > 0:
        recent_ecc = np.mean(eccs[-5:])
        ecc_zscore = abs(recent_ecc - ecc_mean) / ecc_std

        if ecc_zscore > 3.0:
            anomalies.append({
                "type": "ECCENTRICITY_ANOMALY",
                "severity": "HIGH",
                "detail": f"Eccentricity z-score: {ecc_zscore:.1f}",
                "value": recent_ecc,
                "baseline": ecc_mean,
            })
            risk_score += 30
        elif ecc_zscore > 2.0:
            anomalies.append({
                "type": "ECCENTRICITY_DRIFT",
                "severity": "MEDIUM",
                "detail": f"Eccentricity z-score: {ecc_zscore:.1f}",
                "value": recent_ecc,
                "baseline": ecc_mean,
            })
            risk_score += 15

    # --- Mean motion trend analizi ---
    mms = [h["mean_motion"] for h in tle_history]
    mm_mean = np.mean(mms)
    mm_std = np.std(mms) if len(mms) > 1 else 0

    if mm_std > 0:
        recent_mm = np.mean(mms[-5:])
        mm_zscore = abs(recent_mm - mm_mean) / mm_std

        if mm_zscore > 3.0:
            anomalies.append({
                "type": "MEAN_MOTION_JUMP",
                "severity": "HIGH",
                "detail": f"Mean motion z-score: {mm_zscore:.1f} — yükseklik değişimi",
                "value": recent_mm,
                "baseline": mm_mean,
            })
            risk_score += 25
        elif mm_zscore > 2.0:
            anomalies.append({
                "type": "MEAN_MOTION_SHIFT",
                "severity": "MEDIUM",
                "detail": f"Mean motion z-score: {mm_zscore:.1f}",
                "value": recent_mm,
                "baseline": mm_mean,
            })
            risk_score += 10

    # --- Trend yönü (hızlanıyor mu?) ---
    if len(bstars) >= 10:
        first_half = np.mean(bstars[:len(bstars)//2])
        second_half = np.mean(bstars[len(bstars)//2:])
        if first_half != 0:
            trend_ratio = second_half / first_half
            if trend_ratio > 2.0:
                anomalies.append({
                    "type": "ACCELERATING_DECAY",
                    "severity": "HIGH",
                    "detail": f"BSTAR trend 2. yarı / 1. yarı = {trend_ratio:.1f}x",
                    "value": second_half,
                    "baseline": first_half,
                })
                risk_score += 10

    # Risk seviyesi
    risk_score = min(risk_score, 100)
    if risk_score >= 60:
        status = "CRITICAL"
    elif risk_score >= 30:
        status = "WARNING"
    elif risk_score >= 10:
        status = "WATCH"
    else:
        status = "NOMINAL"

    return {
        "object_name": object_name,
        "risk_score": risk_score,
        "status": status,
        "anomalies": anomalies,
        "data_points": len(tle_history),
        "bstar_trend": {
            "mean": float(bstar_mean),
            "std": float(bstar_std),
            "recent": float(np.mean(bstars[-5:])) if len(bstars) >= 5 else float(bstar_mean),
        },
        "eccentricity_trend": {
            "mean": float(ecc_mean),
            "std": float(ecc_std),
            "recent": float(np.mean(eccs[-5:])) if len(eccs) >= 5 else float(ecc_mean),
        },
    }


def scan_breakup_candidates(client, threat_list, top_n=5):
    """
    Tehdit listesindeki roket gövdeleri ve ölü uyduları
    parçalanma riski açısından tarar.
    
    En riskli top_n nesneyi döndürür.
    """
    candidates = []

    # Sadece roket gövdeleri ve enkaz parçalarını tara
    risky_types = ["ROCKET BODY", "DEBRIS"]
    targets = [t for t in threat_list if t.get("object_type") in risky_types]

    print(f"  Taranacak nesne: {len(targets)}")

    for i, target in enumerate(targets[:20]):  # API limiti için max 20
        norad_id = target.get("norad_id", "")
        name = target.get("name", "UNKNOWN")

        if not norad_id:
            continue

        history = fetch_tle_history(client, norad_id)
        if not history:
            continue

        result = analyze_breakup_risk(history, name)
        result["norad_id"] = norad_id
        result["object_type"] = target.get("object_type", "")
        result["country"] = target.get("country", "UNK")
        result["distance_km"] = target.get("distance_km", 0)
        candidates.append(result)

        # Progress
        if (i + 1) % 5 == 0:
            print(f"    ...{i+1}/{len(targets[:20])} tarandı")

    # Risk skoruna göre sırala
    candidates.sort(key=lambda x: x["risk_score"], reverse=True)
    return candidates[:top_n]

# ============================================
# TEST
# ============================================
# ============================================
# FEATURE #17 — FREKANS ÇAKIŞMA RADARI
# ============================================

# Türksat frekans bantları (gerçek veriler)
TURKSAT_FREQUENCIES = {
    "Turksat 3A":  {"bands": ["Ku"], "slots": ["42.0E"], "bw_mhz": 864},
    "Turksat 4A":  {"bands": ["Ku", "C"], "slots": ["42.0E"], "bw_mhz": 1584},
    "Turksat 4B":  {"bands": ["Ku", "C"], "slots": ["50.0E"], "bw_mhz": 1584},
    "Turksat 5A":  {"bands": ["Ku"], "slots": ["31.0E"], "bw_mhz": 756},
    "Turksat 5B":  {"bands": ["Ku", "Ka", "HTS"], "slots": ["42.0E"], "bw_mhz": 2880},
    "Turksat 6A":  {"bands": ["Ku", "C", "Ka"], "slots": ["42.0E"], "bw_mhz": 2160},
}

# GEO'daki bilinen aktif uydu frekans bantları (basitleştirilmiş)
KNOWN_GEO_FREQUENCIES = {
    "INTELSAT":  ["C", "Ku", "Ka"],
    "SES":       ["C", "Ku", "Ka"],
    "EUTELSAT":  ["Ku", "Ka"],
    "ARABSAT":   ["C", "Ku", "Ka"],
    "INMARSAT":  ["L", "C", "Ka"],
    "GORIZONT":  ["C", "Ku"],
    "RADUGA":    ["C", "Ku"],
    "EKRAN":     ["Ku"],
    "COSMOS":    ["C", "Ku"],
    "TDRS":      ["S", "Ku", "Ka"],
}


def check_rf_interference(turksat_name, neighbor_list):
    """
    Türksat uydusu ile komşu uydular arasında
    frekans çakışma riski kontrol eder.
    
    Aynı frekans bandında çalışan ve boylam farkı < 2° olan
    uydular RF girişimi (interference) yaratabilir.
    """
    if turksat_name not in TURKSAT_FREQUENCIES:
        return []

    turksat_bands = set(TURKSAT_FREQUENCIES[turksat_name]["bands"])
    rf_threats = []

    for neighbor in neighbor_list:
        name = neighbor.get("name", "")
        lon_diff = neighbor.get("lon_diff_deg", 999)
        obj_type = neighbor.get("object_type", "")

        # Sadece aktif payload'ları kontrol et (çöp sinyal yaymaz)
        if obj_type != "PAYLOAD":
            continue

        # Komşunun frekans bandını tahmin et
        neighbor_bands = _guess_frequency_bands(name)
        if not neighbor_bands:
            continue

        # Çakışan bantları bul
        overlap = turksat_bands.intersection(neighbor_bands)
        if not overlap:
            continue

        # RF girişim seviyesi (boylam farkına göre)
        if lon_diff < 0.5:
            severity = "CRITICAL"
            interference_db = round(-10 * np.log10(max(lon_diff, 0.01)), 1)
        elif lon_diff < 1.0:
            severity = "HIGH"
            interference_db = round(-10 * np.log10(lon_diff), 1)
        elif lon_diff < 2.0:
            severity = "MEDIUM"
            interference_db = round(-10 * np.log10(lon_diff), 1)
        else:
            severity = "LOW"
            interference_db = round(-10 * np.log10(lon_diff), 1)

        rf_threats.append({
            "neighbor_name": name,
            "lon_diff_deg": lon_diff,
            "overlapping_bands": list(overlap),
            "severity": severity,
            "interference_estimate_db": interference_db,
            "country": neighbor.get("country", "UNK"),
            "distance_km": neighbor.get("distance_km", 0),
        })

    # Ciddiyete göre sırala
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    rf_threats.sort(key=lambda x: severity_order.get(x["severity"], 4))

    return rf_threats


def _guess_frequency_bands(satellite_name):
    """Uydu isminden frekans bantlarını tahmin eder"""
    name_upper = satellite_name.upper()
    for prefix, bands in KNOWN_GEO_FREQUENCIES.items():
        if prefix in name_upper:
            return set(bands)
    # Bilinmeyen uydular için genel tahmin
    if any(kw in name_upper for kw in ["SAT", "STAR", "COM"]):
        return {"C", "Ku"}
    return set()


def full_rf_analysis(neighborhood_results):
    """
    Tüm Türksat uyduları için RF girişim analizi yapar.
    neighborhood_results: geo_neighborhood_analysis() çıktısı
    """
    results = {}

    for sat_name, data in neighborhood_results.items():
        if sat_name not in TURKSAT_FREQUENCIES:
            continue

        rf_threats = check_rf_interference(sat_name, data["neighbors"])

        critical_count = sum(1 for t in rf_threats if t["severity"] == "CRITICAL")
        high_count = sum(1 for t in rf_threats if t["severity"] == "HIGH")

        if critical_count > 0:
            rf_status = "RF_CRITICAL"
        elif high_count > 0:
            rf_status = "RF_WARNING"
        elif len(rf_threats) > 0:
            rf_status = "RF_MONITOR"
        else:
            rf_status = "RF_CLEAR"

        results[sat_name] = {
            "turksat_bands": TURKSAT_FREQUENCIES[sat_name]["bands"],
            "rf_status": rf_status,
            "total_rf_threats": len(rf_threats),
            "critical_count": critical_count,
            "high_count": high_count,
            "threats": rf_threats,
        }

    return results

if __name__ == "__main__":
    from data_fetch import (
        load_all_data, calculate_positions_batch,
        calculate_positions_list, find_closest_threats,
        SpaceTrackClient
    )

    print("=" * 60)
    print("🛰️  YÖRÜNGE KALKANI — TEHDİT ANALİZ TESTİ")
    print("=" * 60)

    # Veri yükle
    turkish, geo_debris, leo_debris = load_all_data(use_cache=True)

    # === TEST 1: Komşuluk Analizi ===
    print("\n🔍 TEST 1: GEO komşuluk analizi")
    tr_positions = calculate_positions_batch(turkish)
    geo_positions = calculate_positions_list(geo_debris)

    neighborhood = geo_neighborhood_analysis(
        tr_positions, turkish,
        geo_positions, geo_debris,
        longitude_range_deg=5.0
    )
    print_neighborhood_report(neighborhood)

    # === TEST 2: Parçalanma Erken Uyarısı ===
    print(f"\n{'='*60}")
    print("💥 TEST 2: Parçalanma erken uyarısı")
    print("-" * 60)

    # Türksat 4A'ya en yakın tehditleri al
    threats_4a = find_closest_threats(tr_positions["Turksat 4A"], geo_positions, n=10)

    # Space-Track'e bağlan (TLE geçmişi için)
    client = SpaceTrackClient()
    if client.login():
        print("  Space-Track bağlantısı ✓")
        print("  Roket gövdeleri ve enkaz taranıyor...\n")

        breakup_results = scan_breakup_candidates(client, threats_4a, top_n=5)

        for r in breakup_results:
            status_icon = {
                "CRITICAL": "🔴", "WARNING": "🟠",
                "WATCH": "🟡", "NOMINAL": "🟢",
                "INSUFFICIENT_DATA": "⚪"
            }.get(r["status"], "⚪")

            print(f"  {status_icon} {r['object_name']:<28} "
                  f"Risk: {r['risk_score']:>3}/100  "
                  f"Durum: {r['status']:<12} "
                  f"[{r['object_type']}] ({r['country']})")

            for a in r["anomalies"]:
                print(f"      ⚠️  {a['type']}: {a['detail']}")
    else:
        print("  ❌ Space-Track bağlantısı başarısız")

    print(f"\n{'='*60}")
    # === TEST 3: Frekans Çakışma ===
    print(f"\n{'='*60}")
    print("📡 TEST 3: RF frekans çakışma analizi")
    print("-" * 60)

    rf_results = full_rf_analysis(neighborhood)

    for sat_name, rf in rf_results.items():
        status_icon = {
            "RF_CRITICAL": "🔴", "RF_WARNING": "🟠",
            "RF_MONITOR": "🟡", "RF_CLEAR": "🟢"
        }.get(rf["rf_status"], "⚪")

        print(f"\n  {status_icon} {sat_name} [{', '.join(rf['turksat_bands'])}]")
        print(f"     Durum: {rf['rf_status']}  |  Tehdit: {rf['total_rf_threats']}")

        for t in rf["threats"][:3]:
            sev_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(t["severity"], "⚪")
            print(f"     {sev_icon} {t['neighbor_name']:<25} "
                  f"Δ={t['lon_diff_deg']:.2f}° "
                  f"Bant: {', '.join(t['overlapping_bands'])} "
                  f"~{t['interference_estimate_db']}dB "
                  f"({t['country']})")
    print("✅ Tehdit analiz testleri tamamlandı!")