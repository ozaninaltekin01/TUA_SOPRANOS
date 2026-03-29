# ============================================
# YÖRÜNGE KALKANI — YÖRÜNGE HESAPLAMA MOTORU
# ============================================

import numpy as np
import datetime
from sgp4.api import Satrec, jday
from config import EARTH_RADIUS_KM, COVARIANCE_PARAMS
from data_fetch import (
    calculate_position, generate_covariance,
    estimate_hbr, tle_confidence_score
)


# ============================================
# 1. RIC KOORDİNAT DÖNÜŞÜMÜ (CARA STANDARDI)
# ============================================

def eci_to_ric(pos_primary, vel_primary, pos_secondary, vel_secondary):
    """
    ECI (Earth-Centered Inertial) koordinatlarını
    RIC (Radial, In-track, Cross-track) koordinatlarına dönüştürür.
    NASA CARA bu koordinat sistemini kullanır.
    """
    r_p = np.array(pos_primary)
    v_p = np.array(vel_primary)
    r_s = np.array(pos_secondary)

    # RIC birim vektörleri
    R_hat = r_p / np.linalg.norm(r_p)                    # Radial
    C_hat = np.cross(r_p, v_p)                            # Cross-track
    C_hat = C_hat / np.linalg.norm(C_hat)
    I_hat = np.cross(C_hat, R_hat)                        # In-track

    # Dönüşüm matrisi
    T = np.array([R_hat, I_hat, C_hat])

    # Relative pozisyon (RIC'de)
    delta_r = r_s - r_p
    ric_pos = T @ delta_r

    # Relative hız (RIC'de)
    v_s = np.array(vel_secondary)
    delta_v = v_s - v_p
    ric_vel = T @ delta_v

    return {
        "ric_pos": ric_pos.tolist(),       # [R, I, C] km
        "ric_vel": ric_vel.tolist(),       # [vR, vI, vC] km/s
        "rotation_matrix": T.tolist(),
        "miss_distance_km": float(np.linalg.norm(delta_r)),
        "radial_km": float(ric_pos[0]),
        "intrack_km": float(ric_pos[1]),
        "crosstrack_km": float(ric_pos[2]),
    }


def transform_covariance_to_ric(cov_eci, rotation_matrix):
    """Kovaryans matrisini ECI'den RIC'e dönüştürür"""
    T = np.array(rotation_matrix)
    C = np.array(cov_eci)
    return (T @ C @ T.T).tolist()


# ============================================
# 2. TCA HESABI (Time of Closest Approach)
# ============================================

def compute_tca(line1_p, line2_p, line1_s, line2_s,
                hours_ahead=24, step_minutes=1, fine_step_seconds=None):
    """
    İki nesnenin en yakın geçiş zamanını (TCA) ve
    o andaki mesafeyi hesaplar.

    İki aşamalı arama:
    1. Kaba tarama (step_minutes aralıklarla)
    2. İnce tarama (±1 dakika etrafında)
       - LEO (fine_step_seconds=1): 7.5 km/s hız → 1 sn = 7.5 km hassasiyet
       - GEO (fine_step_seconds=6): 3 km/s hız → 6 sn = 18 km hassasiyet

    fine_step_seconds=None → otomatik belirlenir (LEO için 1, diğerleri için 6)
    """
    sat_p = Satrec.twoline2rv(line1_p, line2_p)
    sat_s = Satrec.twoline2rv(line1_s, line2_s)

    now = datetime.datetime.utcnow()
    min_dist = float('inf')
    tca_time = now
    tca_r_p = tca_v_p = tca_r_s = tca_v_s = None

    # --- Aşama 1: Kaba tarama ---
    best_minute = 0
    for minute in range(int(hours_ahead * 60)):
        t = now + datetime.timedelta(minutes=minute)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)

        e1, r_p, v_p = sat_p.sgp4(jd, fr)
        e2, r_s, v_s = sat_s.sgp4(jd, fr)

        if e1 != 0 or e2 != 0:
            continue
        if any(np.isnan(r_p)) or any(np.isnan(r_s)):
            continue

        dist = np.sqrt(sum((a - b)**2 for a, b in zip(r_p, r_s)))

        if dist < min_dist:
            min_dist = dist
            best_minute = minute
            tca_time = t
            tca_r_p, tca_v_p = list(r_p), list(v_p)
            tca_r_s, tca_v_s = list(r_s), list(v_s)

    # --- Aşama 2: İnce tarama (±1 dakika) ---
    # LEO'da 7.5 km/s hız → 6 sn adımda 45 km atlayabilir, 1 sn kullan
    # GEO'da 3 km/s hız → 6 sn yeterli
    if fine_step_seconds is None:
        speeds = [np.linalg.norm(tca_v_p), np.linalg.norm(tca_v_s)]
        fine_step_seconds = 1 if max(speeds) > 5.0 else 6
    fine_start = max(0, best_minute - 1)
    fine_end = best_minute + 1
    for second in range(fine_start * 60, fine_end * 60, fine_step_seconds):
        t = now + datetime.timedelta(seconds=second)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)

        e1, r_p, v_p = sat_p.sgp4(jd, fr)
        e2, r_s, v_s = sat_s.sgp4(jd, fr)

        if e1 != 0 or e2 != 0:
            continue
        if any(np.isnan(r_p)) or any(np.isnan(r_s)):
            continue

        dist = np.sqrt(sum((a - b)**2 for a, b in zip(r_p, r_s)))

        if dist < min_dist:
            min_dist = dist
            tca_time = t
            tca_r_p, tca_v_p = list(r_p), list(v_p)
            tca_r_s, tca_v_s = list(r_s), list(v_s)

    # Sonuç
    time_to_tca = (tca_time - now).total_seconds()

    return {
        "tca_utc": tca_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "time_to_tca_seconds": round(time_to_tca),
        "time_to_tca_hours": round(time_to_tca / 3600, 2),
        "min_distance_km": round(min_dist, 3),
        "primary_pos": tca_r_p,
        "primary_vel": tca_v_p,
        "secondary_pos": tca_r_s,
        "secondary_vel": tca_v_s,
    }


# ============================================
# 3. ÇARPIŞMA DÜZLEMİ PROJEKSİYONU (CARA 2D Pc için)
# ============================================

def compute_conjunction_plane(tca_result):
    """
    TCA anındaki verileri kullanarak çarpışma düzlemi projeksiyonu yapar.
    Bu çıktı doğrudan K2'nin CARA Pc hesabına girdi olur.
    """
    r_p = np.array(tca_result["primary_pos"])
    v_p = np.array(tca_result["primary_vel"])
    r_s = np.array(tca_result["secondary_pos"])
    v_s = np.array(tca_result["secondary_vel"])

    # Göreli hız ve pozisyon
    rel_vel = v_s - v_p
    rel_pos = r_s - r_p

    rel_speed = np.linalg.norm(rel_vel)
    miss_distance = np.linalg.norm(rel_pos)

    # Çarpışma düzlemi normal vektörü (göreli hız yönü)
    if rel_speed < 1e-10:
        return None

    n_hat = rel_vel / rel_speed

    # Çarpışma düzlemi koordinat sistemi
    # u1: miss distance yönü (düzlemde)
    # u2: düzlemin ikinci ekseni
    rel_pos_perp = rel_pos - np.dot(rel_pos, n_hat) * n_hat
    perp_norm = np.linalg.norm(rel_pos_perp)

    if perp_norm < 1e-10:
        u1 = np.array([1, 0, 0]) - np.dot(np.array([1, 0, 0]), n_hat) * n_hat
        u1 = u1 / np.linalg.norm(u1)
    else:
        u1 = rel_pos_perp / perp_norm

    u2 = np.cross(n_hat, u1)

    # Projeksiyon matrisi (3D → 2D çarpışma düzlemi)
    P = np.array([u1, u2])  # 2x3

    # Miss distance vektörünü çarpışma düzlemine projekte et
    miss_2d = P @ rel_pos  # 2x1

    return {
        "miss_2d": miss_2d.tolist(),           # [x, y] km (düzlemde)
        "miss_distance_km": float(miss_distance),
        "relative_speed_kms": float(rel_speed),
        "projection_matrix": P.tolist(),        # 2x3 matris
        "normal_vector": n_hat.tolist(),
        "u1": u1.tolist(),
        "u2": u2.tolist(),
    }


def project_covariance_to_2d(cov_3d_primary, cov_3d_secondary, projection_matrix):
    """
    İki nesnenin 3D kovaryans matrislerini çarpışma düzlemine
    projekte eder → 2D birleşik kovaryans (CARA Pc hesabı için)
    """
    P = np.array(projection_matrix)  # 2x3
    C1 = np.array(cov_3d_primary)    # 3x3
    C2 = np.array(cov_3d_secondary)  # 3x3

    # Birleşik kovaryans (bağımsız nesneler varsayımı)
    C_combined = C1 + C2

    # 2D'ye projeksiyon: C_2d = P @ C_combined @ P^T
    C_2d = P @ C_combined @ P.T  # 2x2

    return {
        "combined_covariance_2d": C_2d.tolist(),
        "combined_covariance_3d": C_combined.tolist(),
        "det": float(np.linalg.det(C_2d)),
        "eigenvalues": np.linalg.eigvalsh(C_2d).tolist(),
    }


# ============================================
# 4. TAM CONJUNCTION ANALİZİ (K2'ye hazır paket)
# ============================================

def full_conjunction_analysis(primary_tle, secondary_tle,
                               primary_info, secondary_info,
                               hours_ahead=24):
    """
    Tek fonksiyonda tam conjunction analizi:
    TCA → RIC → çarpışma düzlemi → kovaryans projeksiyonu
    
    Çıktı doğrudan K2'nin cara_engine.py dosyasına girdi olarak gider.
    """

    # 1. TCA hesapla
    tca = compute_tca(
        primary_tle["line1"], primary_tle["line2"],
        secondary_tle["line1"], secondary_tle["line2"],
        hours_ahead=hours_ahead
    )

    if tca["primary_pos"] is None:
        return None

    # 2. RIC dönüşümü
    ric = eci_to_ric(
        tca["primary_pos"], tca["primary_vel"],
        tca["secondary_pos"], tca["secondary_vel"]
    )

    # 3. Çarpışma düzlemi
    conj_plane = compute_conjunction_plane(tca)
    if conj_plane is None:
        return None

    # 4. Kovaryans matrisleri
    p_orbit = primary_info.get("orbit", "GEO")
    s_orbit = secondary_info.get("orbit", "LEO") if "orbit" in secondary_info else "LEO"

    cov_p, sigma_p, age_p = generate_covariance(primary_tle.get("epoch", ""), p_orbit)
    cov_s, sigma_s, age_s = generate_covariance(secondary_tle.get("epoch", ""), s_orbit)

    # 5. 2D kovaryans projeksiyonu
    cov_2d = project_covariance_to_2d(
        cov_p, cov_s,
        conj_plane["projection_matrix"]
    )

    # 6. HBR hesabı
    hbr_p = estimate_hbr(
        primary_info.get("rcs_size", "LARGE"),
        primary_info.get("mass_kg", None)
    )
    hbr_s = estimate_hbr(
        secondary_info.get("rcs_size", "MEDIUM"),
        secondary_info.get("mass_kg", None)
    )
    combined_hbr = hbr_p + hbr_s

    # 7. TLE güven skorları
    conf_p, age_h_p = tle_confidence_score(primary_tle.get("epoch", ""), p_orbit)
    conf_s, age_h_s = tle_confidence_score(secondary_tle.get("epoch", ""), s_orbit)

    return {
        # TCA bilgileri
        "tca_utc": tca["tca_utc"],
        "time_to_tca_hours": tca["time_to_tca_hours"],
        "time_to_tca_seconds": tca["time_to_tca_seconds"],
        "min_distance_km": tca["min_distance_km"],

        # RIC bileşenleri
        "radial_km": ric["radial_km"],
        "intrack_km": ric["intrack_km"],
        "crosstrack_km": ric["crosstrack_km"],

        # Çarpışma düzlemi
        "miss_2d": conj_plane["miss_2d"],
        "relative_speed_kms": conj_plane["relative_speed_kms"],
        "projection_matrix": conj_plane["projection_matrix"],

        # Kovaryans (CARA Pc hesabı için)
        "combined_covariance_2d": cov_2d["combined_covariance_2d"],
        "covariance_det": cov_2d["det"],

        # HBR
        "hbr_primary_m": round(hbr_p, 2),
        "hbr_secondary_m": round(hbr_s, 2),
        "combined_hbr_m": round(combined_hbr, 2),
        "combined_hbr_km": round(combined_hbr / 1000, 6),

        # Güven
        "confidence_primary": conf_p,
        "confidence_secondary": conf_s,
        "confidence_combined": round((conf_p + conf_s) / 2, 1),

        # Pozisyonlar (gerekirse)
        "primary_pos_tca": tca["primary_pos"],
        "primary_vel_tca": tca["primary_vel"],
        "secondary_pos_tca": tca["secondary_pos"],
        "secondary_vel_tca": tca["secondary_vel"],
    }


# ============================================
# TEST
# ============================================

if __name__ == "__main__":
    from data_fetch import load_all_data, find_closest_threats, calculate_positions_batch, calculate_positions_list

    print("=" * 60)
    print("🛰️  YÖRÜNGE KALKANI — CONJUNCTION ANALİZ TESTİ")
    print("=" * 60)

    # Veri yükle (cache varsa oradan)
    turkish, geo_debris, leo_debris = load_all_data(use_cache=True)

    # Test: Turksat 4A vs en yakın GEO tehdidi
    test_primary = "Turksat 4A"
    print(f"\n🎯 Test: {test_primary} conjunction analizi")
    print("-" * 60)

    # Pozisyonları hesapla
    tr_pos = calculate_positions_batch(turkish)
    geo_pos = calculate_positions_list(geo_debris)

    # En yakın tehdidi bul
    threats = find_closest_threats(tr_pos[test_primary], geo_pos, n=1)

    if not threats:
        print("❌ Tehdit bulunamadı")
    else:
        threat = threats[0]
        print(f"  Tehdit: {threat['name']} ({threat['distance_km']:.1f} km)")

        # Tehdidin TLE'sini bul
        threat_tle = None
        for obj in geo_debris:
            if obj["object_name"] == threat["name"]:
                threat_tle = obj
                break

        if threat_tle:
            print(f"\n⏱️  TCA hesaplanıyor (24 saat ileri)...")

            result = full_conjunction_analysis(
                primary_tle=turkish[test_primary],
                secondary_tle=threat_tle,
                primary_info={"orbit": "GEO", "mass_kg": 4869, "rcs_size": "LARGE"},
                secondary_info={"orbit": "GEO", "rcs_size": threat.get("rcs_size", "MEDIUM")},
                hours_ahead=24
            )

            if result:
                print(f"\n📊 CONJUNCTION ANALİZ SONUÇLARI")
                print(f"  TCA Zamanı      : {result['tca_utc']}")
                print(f"  TCA'ya Kalan    : {result['time_to_tca_hours']:.1f} saat")
                print(f"  Min Mesafe      : {result['min_distance_km']:.3f} km")
                print(f"  ")
                print(f"  RIC Bileşenleri:")
                print(f"    Radial        : {result['radial_km']:.3f} km")
                print(f"    In-track      : {result['intrack_km']:.3f} km")
                print(f"    Cross-track   : {result['crosstrack_km']:.3f} km")
                print(f"  ")
                print(f"  Göreli Hız      : {result['relative_speed_kms']:.4f} km/s")
                print(f"  Miss 2D         : [{result['miss_2d'][0]:.3f}, {result['miss_2d'][1]:.3f}] km")
                print(f"  ")
                print(f"  HBR Birleşik    : {result['combined_hbr_m']:.2f} m ({result['combined_hbr_km']:.6f} km)")
                print(f"  Kovaryans Det   : {result['covariance_det']:.6f}")
                print(f"  ")
                print(f"  TLE Güven:")
                print(f"    Primary       : {result['confidence_primary']}%")
                print(f"    Secondary     : {result['confidence_secondary']}%")
                print(f"    Birleşik      : {result['confidence_combined']}%")
                print(f"\n  ✅ Bu çıktı K2'nin cara_engine.py dosyasına hazır!")
            else:
                print("❌ Conjunction analizi başarısız")

    # Test 2: Göktürk-1 (LEO) analizi
    test_leo = "Gokturk-1"
    if test_leo in turkish and test_leo in tr_pos:
        print(f"\n{'=' * 60}")
        print(f"🎯 Test: {test_leo} (LEO) conjunction analizi")
        print("-" * 60)

        leo_pos = calculate_positions_list(leo_debris)
        threats_leo = find_closest_threats(tr_pos[test_leo], leo_pos, n=1)

        if threats_leo:
            threat_l = threats_leo[0]
            print(f"  Tehdit: {threat_l['name']} ({threat_l['distance_km']:.1f} km)")

            threat_tle_l = None
            for obj in leo_debris:
                if obj["object_name"] == threat_l["name"]:
                    threat_tle_l = obj
                    break

            if threat_tle_l:
                print(f"\n⏱️  TCA hesaplanıyor...")

                result_l = full_conjunction_analysis(
                    primary_tle=turkish[test_leo],
                    secondary_tle=threat_tle_l,
                    primary_info={"orbit": "LEO", "mass_kg": 1060, "rcs_size": "LARGE"},
                    secondary_info={"orbit": "LEO", "rcs_size": threat_l.get("rcs_size", "MEDIUM")},
                    hours_ahead=12  # LEO için 12 saat yeterli
                )

                if result_l:
                    print(f"\n📊 LEO CONJUNCTION SONUÇLARI")
                    print(f"  TCA Zamanı      : {result_l['tca_utc']}")
                    print(f"  TCA'ya Kalan    : {result_l['time_to_tca_hours']:.1f} saat")
                    print(f"  Min Mesafe      : {result_l['min_distance_km']:.3f} km")
                    print(f"  Göreli Hız      : {result_l['relative_speed_kms']:.4f} km/s")
                    print(f"  HBR Birleşik    : {result_l['combined_hbr_m']:.2f} m")
                    print(f"  Güven           : {result_l['confidence_combined']}%")

    print(f"\n{'=' * 60}")
    print("✅ Conjunction analiz testi tamamlandı!")
    print("📦 K2'ye teslim edilecek fonksiyon: full_conjunction_analysis()")