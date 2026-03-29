"""
run_k2_live.py — K1 + K2 Gerçek Veri Entegrasyonu
=====================================================
K3 olmadan, terminalde K1'in gerçek verisini K2'ye besler.

Kullanım:
    cd tua_sopranos1
    python run_k2_live.py

Gereksinimler:
    - Veri_analizi/ klasörü (K1) repo'da olmalı
    - model/ klasörü (K2) repo'da olmalı
    - pip install numpy scipy sgp4 requests
    - Space-Track hesabı (config.py'de tanımlı)

İlk çalıştırmada Space-Track'ten veri çeker ve cache'ler.
Sonraki çalıştırmalarda cache'ten okur (hızlı).
"""

import sys
import os
import time

# Path ayarla — repo kök dizininden çalışmalı
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Veri_analizi"))

# ============================================================
# ADIM 0: MODÜL KONTROL
# ============================================================

print("=" * 65)
print("  K1 + K2 CANLI ENTEGRASYON")
print("  TUA SOPRANOS — Mock Data Yok, Gerçek Veri")
print("=" * 65)

print("\n📦 Modül kontrolü...")

try:
    from Veri_analizi.config import (
        TURKISH_SATELLITES, CARA_THRESHOLDS as K1_THRESHOLDS,
        SATELLITE_OPERATIONS,
    )
    print("  ✅ Veri_analizi.config")
except ImportError as e:
    print(f"  ❌ Veri_analizi.config — {e}")
    print("\n  Veri_analizi/ klasörü bulunamadı!")
    print("  Emin ol ki repo yapısı şöyle:")
    print("    tua_sopranos1/")
    print("    ├── Veri_analizi/")
    print("    │   ├── config.py")
    print("    │   ├── data_fetch.py")
    print("    │   └── orbit_calc.py")
    print("    ├── model/")
    print("    └── run_k2_live.py  ← BU DOSYA")
    sys.exit(1)

try:
    from Veri_analizi.data_fetch import (
        load_all_data, calculate_positions_batch, calculate_positions_list,
        find_closest_threats, generate_covariance, estimate_hbr,
        tle_confidence_score
    )
    print("  ✅ Veri_analizi.data_fetch")
except ImportError as e:
    print(f"  ❌ Veri_analizi.data_fetch — {e}")
    sys.exit(1)

try:
    from Veri_analizi.orbit_calc import full_conjunction_analysis
    print("  ✅ Veri_analizi.orbit_calc")
except ImportError as e:
    print(f"  ❌ Veri_analizi.orbit_calc — {e}")
    sys.exit(1)

try:
    from model.cara_engine import run_cara_from_k1, batch_cara_assessment, generate_cdm
    from model.maneuver import suggest_maneuver, shadow_zone_analysis
    from model.game_theory import who_should_dodge, fuel_budget_manager
    from model.threat_analysis import detect_ghost_maneuver, fragmentation_warning, prioritize_threats
    print("  ✅ model (K2 tüm modüller)")
except ImportError as e:
    print(f"  ❌ model — {e}")
    sys.exit(1)

try:
    from model.ml_model import predict_risk, load_model
    from model.model_evaluation import quick_sanity_check
    _xgb_model_data = load_model()
    print("  ✅ model.ml_model (XGBoost)")
except Exception:
    _xgb_model_data = None
    print("  ⚠️  model.ml_model — XGBoost yüklenemedi (screening devre dışı)")

print("\n  Tüm modüller yüklendi!")

# ── Model sağlık kontrolü ──────────────────────────────────────────────────
if _xgb_model_data is not None:
    try:
        sc = quick_sanity_check()
        xgb_status = "✅" if sc["xgboost_ok"] else "⚠️ "
        print(f"  {xgb_status} XGBoost sanity: {sc['details'].get('xgboost', 'ok')}")
    except Exception:
        pass


# ============================================================
# ADIM 1: K1'DEN VERİ ÇEK
# ============================================================

print(f"\n{'=' * 65}")
print("  ADIM 1: VERİ YÜKLEME (K1)")
print(f"{'=' * 65}")

# İlk seferde API'den çeker, sonra cache kullanır
print("\n📡 Veri yükleniyor (cache varsa oradan)...")
start = time.time()

try:
    turkish, geo_debris, leo_debris = load_all_data(use_cache=True)
except Exception as e:
    print(f"\n⚠️ Cache bulunamadı, API'den çekiliyor...")
    try:
        turkish, geo_debris, leo_debris = load_all_data(use_cache=False)
    except Exception as e2:
        print(f"❌ Veri yüklenemedi: {e2}")
        print("   Space-Track login bilgilerini config.py'de kontrol et.")
        sys.exit(1)

elapsed = time.time() - start
print(f"\n  Yükleme süresi: {elapsed:.1f} saniye")
print(f"  Türk uydusu: {len(turkish)}")
print(f"  GEO çöp: {len(geo_debris)}")
print(f"  LEO çöp: {len(leo_debris)}")


# ============================================================
# ADIM 2: POZİSYON HESAPLA (K1)
# ============================================================

print(f"\n{'=' * 65}")
print("  ADIM 2: POZİSYON HESABI (K1 — SGP4)")
print(f"{'=' * 65}")

print("\n📍 Türk uyduları pozisyonları hesaplanıyor...")
tr_positions = calculate_positions_batch(turkish)

for name, pos in tr_positions.items():
    orbit = turkish[name]["orbit"]
    conf, age = tle_confidence_score(turkish[name]["epoch"], orbit)
    print(f"  {name:<16} alt={pos['altitude_km']:>10.1f} km  "
          f"hız={pos['speed_kms']:.3f} km/s  "
          f"güven={conf}% ({age:.0f}h)")

print(f"\n🗑️  Çöp pozisyonları hesaplanıyor...")
geo_positions = calculate_positions_list(geo_debris)
leo_positions = calculate_positions_list(leo_debris)
print(f"  GEO: {len(geo_positions)} nesne")
print(f"  LEO: {len(leo_positions)} nesne")


# ============================================================
# ADIM 3: TEHDİT TARAMA (K1) + ÖNCELİKLENDİRME (K2)
# ============================================================

print(f"\n{'=' * 65}")
print("  ADIM 3: TEHDİT TARAMA + ÖNCELİKLENDİRME")
print(f"{'=' * 65}")

all_threats = {}

for sat_name, sat_pos in tr_positions.items():
    orbit = turkish[sat_name]["orbit"]
    debris_pool = geo_positions if orbit == "GEO" else leo_positions
    
    threats = find_closest_threats(sat_pos, debris_pool, n=5,
                                    exclude_names=[sat_name])
    
    # K2 önceliklendirme
    prioritized = prioritize_threats(threats)
    all_threats[sat_name] = prioritized
    
    print(f"\n  🎯 {sat_name} ({orbit}) — En yakın 5 tehdit:")
    for t in prioritized:
        print(f"     {t['priority_rank']}. {t['name']:<25} "
              f"{t['distance_km']:>8.1f} km  "
              f"[{t.get('rcs_size', '?')}]  "
              f"skor={t.get('threat_score', 0)}")


# ============================================================
# ADIM 3.5: XGBOOST HIZLI SCREENING (K2)
# Tüm üst tehditleri milisaniyeler içinde sınıflandır.
# Sadece YELLOW/RED çıkanlar tam CARA Pc hesabına geçer.
# ============================================================

xgb_verdicts = {}   # sat_name → {"threat_name": verdict}

if _xgb_model_data is not None:
    print(f"\n{'=' * 65}")
    print("  ADIM 3.5: XGBOOST HIZLI SCREENING")
    print(f"{'=' * 65}")

    for sat_name, threats in all_threats.items():
        if not threats:
            continue
        orbit = turkish[sat_name]["orbit"]
        xgb_verdicts[sat_name] = {}

        for threat in threats:
            # Hızlı özellik çıkarımı — tam TCA olmadan
            quick_conj = {
                "min_distance_km":      threat["distance_km"],
                "relative_speed_kms":   threat.get("rel_vel", 7.5 if orbit == "LEO" else 0.5),
                "secondary_rcs_m2":     {"LARGE": 10.0, "MEDIUM": 1.0, "SMALL": 0.1}.get(
                                            threat.get("rcs_size", "MEDIUM"), 1.0),
                "time_to_tca_hours":    12.0 if orbit == "LEO" else 24.0,
                "tle_age_hours":        12.0,
            }
            result = predict_risk(quick_conj, _xgb_model_data)
            xgb_verdicts[sat_name][threat["name"]] = result

        top = threats[0]
        v   = xgb_verdicts[sat_name].get(top["name"], {})
        icon = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(
            v.get("predicted_class", "?"), "⚪")
        print(f"  {sat_name:<16} → {top['name']:<25} "
              f"{icon} {v.get('predicted_class','?'):6}  "
              f"güven=%{v.get('confidence_pct', 0):.0f}")
else:
    print("\n  ⚠️  ADIM 3.5 atlandı — XGBoost modeli yok")


# ============================================================
# ADIM 4: CONJUNCTION ANALİZİ (K1) + CARA Pc (K2)
# XGBoost GREEN derse tam CARA atlanır; YELLOW/RED ise hesaplanır.
# ============================================================

print(f"\n{'=' * 65}")
print("  ADIM 4: CONJUNCTION ANALİZİ + CARA Pc HESABI")
print(f"{'=' * 65}")

cara_results = {}

for sat_name, threats in all_threats.items():
    if not threats:
        continue
    
    orbit = turkish[sat_name]["orbit"]
    sat_info = {
        "orbit": orbit,
        "mass_kg": turkish[sat_name].get("mass_kg", 1000),
        "rcs_size": turkish[sat_name].get("rcs_size", "LARGE"),
    }
    
    # XGBoost GREEN diyorsa ve başka YELLOW/RED tehdit yoksa atla
    sat_verdicts = xgb_verdicts.get(sat_name, {})
    non_green = [t for t in threats
                 if sat_verdicts.get(t["name"], {}).get("predicted_class", "GREEN") != "GREEN"]
    threat = non_green[0] if non_green else threats[0]  # en öncelikli tehdit

    # XGBoost GREEN ve model aktifse uyar ama yine de hesapla (doğrulama için)
    if _xgb_model_data and not non_green:
        xgb_conf = sat_verdicts.get(threat["name"], {}).get("confidence_pct", 0)
        print(f"\n  ✅ {sat_name}: XGBoost GREEN (%{xgb_conf:.0f}) — "
              f"CARA doğrulaması yapılıyor...")
    
    # Tehdidin TLE'sini bul
    debris_list = geo_debris if orbit == "GEO" else leo_debris
    threat_tle = None
    for obj in debris_list:
        if obj["object_name"] == threat["name"]:
            threat_tle = obj
            break
    
    if threat_tle is None:
        print(f"\n  ⚠️ {sat_name}: Tehdit TLE bulunamadı — atlanıyor")
        continue
    
    threat_info = {
        "orbit": orbit,
        "rcs_size": threat.get("rcs_size", "MEDIUM"),
    }
    
    print(f"\n  ⏱️  {sat_name} vs {threat['name']}...")
    print(f"     Anlık mesafe: {threat['distance_km']:.1f} km")
    
    try:
        # K1: Tam conjunction analizi (TCA + çarpışma düzlemi + kovaryans)
        hours = 12 if orbit == "LEO" else 24
        conj = full_conjunction_analysis(
            primary_tle=turkish[sat_name],
            secondary_tle=threat_tle,
            primary_info=sat_info,
            secondary_info=threat_info,
            hours_ahead=hours
        )
        
        if conj is None:
            print(f"     ⚠️ Conjunction analizi başarısız")
            continue
        
        print(f"     TCA: {conj['tca_utc']}")
        print(f"     TCA'daki mesafe: {conj['min_distance_km']:.3f} km")
        print(f"     Göreli hız: {conj['relative_speed_kms']:.4f} km/s")
        
        # K2: CARA Pc hesabı
        result = run_cara_from_k1(conj)
        
        status_icon = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(
            result["cara_status"], "⚪"
        )
        
        print(f"     ───────────────────────────────")
        print(f"     {status_icon} CARA: {result['cara_status']}")
        print(f"     Pc: {result['pc_scientific']}")
        print(f"     Pc (log10): {result['pc_log10']}")
        print(f"     Aksiyon: {result['cara_action']}")
        print(f"     TLE Güven: {conj['confidence_combined']}%")
        
        cara_results[sat_name] = {
            "conjunction": conj,
            "cara": result,
            "threat": threat,
            "threat_tle": threat_tle,
        }
        
    except Exception as e:
        print(f"     ❌ Hata: {e}")


# ============================================================
# ADIM 5: MANEVRA ÖNERİSİ (K2) — En tehlikeli uydu için
# ============================================================

print(f"\n{'=' * 65}")
print("  ADIM 5: MANEVRA ÖNERİSİ (En yüksek Pc)")
print(f"{'=' * 65}")

if cara_results:
    # En yüksek Pc'yi bul
    worst = max(cara_results.items(), key=lambda x: x[1]["cara"]["pc"])
    worst_name = worst[0]
    worst_data = worst[1]
    
    print(f"\n  🚨 En yüksek risk: {worst_name}")
    print(f"     Tehdit: {worst_data['threat']['name']}")
    print(f"     Pc: {worst_data['cara']['pc_scientific']}")
    print(f"     CARA: {worst_data['cara']['cara_status']}")
    
    # Manevra önerisi
    mass  = turkish[worst_name].get("mass_kg", 4229)
    orbit = turkish[worst_name]["orbit"]
    ops   = SATELLITE_OPERATIONS.get(worst_name, {"fuel_kg": 100, "mission_years_remaining": 5})
    fuel_kg      = ops["fuel_kg"]
    mission_yrs  = ops["mission_years_remaining"]

    maneuver_result = suggest_maneuver(
        conjunction_data=worst_data["conjunction"],
        spacecraft_mass_kg=mass,
        fuel_remaining_kg=fuel_kg,
        mission_remaining_years=mission_yrs,
        orbit_type=orbit
    )
    
    print(f"\n  📋 Manevra Seçenekleri:")
    for i, opt in enumerate(maneuver_result["options"]):
        marker = " ★" if i == maneuver_result["recommended_index"] else ""
        print(f"\n     Seçenek {i+1}: {opt['level_tr']}{marker}")
        print(f"       Delta-V:      {opt['delta_v_ms']:.4f} m/s")
        print(f"       Yakıt:        {opt['fuel_mass_kg']:.4f} kg")
        print(f"       Maliyet:      ${opt['fuel_cost_usd']:,.2f}")
        print(f"       Ömür kaybı:   {opt['lifetime_loss_days']} gün")
        print(f"       Pc önce:      {opt['pc_before']}")
        print(f"       Pc sonra:     {opt['pc_after']}")
        print(f"       Yeni durum:   {opt['new_cara_status']}")
    
    
    # ============================================================
    # ADIM 6: KİM KAÇINMALI? (K2 — Oyun Teorisi)
    # ============================================================
    
    print(f"\n{'=' * 65}")
    print("  ADIM 6: OYUN TEORİSİ — KİM KAÇINMALI?")
    print(f"{'=' * 65}")
    
    is_debris = worst_data["threat"].get("object_type", "").upper() in [
        "DEBRIS", "ROCKET BODY", "UNKNOWN", "TBA"
    ]
    
    game = who_should_dodge(
        pc=worst_data["cara"]["pc"],
        fuel_remaining_primary_kg=fuel_kg,
        fuel_remaining_secondary_kg=0 if is_debris else 50,
        fuel_cost_primary_kg=maneuver_result["recommended"]["fuel_mass_kg"],
        fuel_cost_secondary_kg=0 if is_debris else 0.8,
        mass_primary_kg=mass,
        mass_secondary_kg=50 if is_debris else 1000,
        is_secondary_debris=is_debris
    )
    
    print(f"\n  🎮 Karar: {game['decision_tr']}")
    print(f"     Sebep: {game['reason']}")
    print(f"     Güven: %{game['confidence']}")
    
    if game.get("game_theory_applicable") and game.get("nash_equilibrium"):
        nash = game["nash_equilibrium"]
        if nash["pure_nash_equilibria"]:
            for ne in nash["pure_nash_equilibria"]:
                print(f"     Nash: Biz={ne['primary_strategy']}, "
                      f"Onlar={ne['secondary_strategy']}")
    
    
    # ============================================================
    # ADIM 7: YAKIT BÜTÇE ANALİZİ (K2)
    # ============================================================
    
    print(f"\n{'=' * 65}")
    print("  ADIM 7: YAKIT BÜTÇE ANALİZİ")
    print(f"{'=' * 65}")
    
    budget = fuel_budget_manager(
        fuel_remaining_kg=fuel_kg,
        mission_remaining_years=mission_yrs,
        annual_conjunctions=15 if orbit == "LEO" else 5,
        orbit_type=orbit
    )
    
    print(f"\n  ⛽ {worst_name} Bütçe Durumu:")
    print(f"     Kalan yakıt:             {budget['fuel_remaining_kg']} kg")
    print(f"     Kalan görev:             {budget['mission_remaining_years']} yıl")
    print(f"     Beklenen tehdit (toplam): {budget['total_expected_conjunctions']}")
    print(f"     Yapılabilir manevra:      {budget['maneuvers_possible']}")
    print(f"     Yeterlilik oranı:         {budget['fuel_sufficiency_ratio']}x")
    print(f"     Optimal Pc eşiği:         {budget['optimal_pc_threshold_scientific']}")
    print(f"     Strateji:                 {budget['strategy']}")
    
    
    # ============================================================
    # ADIM 8: CDM OLUŞTUR (K2)
    # ============================================================
    
    print(f"\n{'=' * 65}")
    print("  ADIM 8: CDM OLUŞTURMA (CCSDS 508.0)")
    print(f"{'=' * 65}")
    
    cdm_xml = generate_cdm(
        primary_name=worst_name,
        secondary_name=worst_data["threat"]["name"],
        assessment=worst_data["cara"],
        primary_norad_id=str(turkish[worst_name].get("norad_id", "99999")),
        secondary_norad_id=str(worst_data["threat"].get("norad_id", "99999")),
    )
    
    # CDM dosyası kaydet
    cdm_filename = f"CDM_{worst_name.replace(' ', '_')}.xml"
    with open(cdm_filename, "w", encoding="utf-8") as f:
        f.write(cdm_xml)
    
    print(f"\n  📄 CDM dosyası oluşturuldu: {cdm_filename}")
    print(f"     Boyut: {len(cdm_xml)} karakter")
    print(f"     İlk 3 satır:")
    for line in cdm_xml.strip().split("\n")[:3]:
        print(f"       {line}")

else:
    print("\n  ⚠️ Hiçbir uydu için conjunction analizi yapılamadı.")
    print("     Cache boş olabilir — önce K1'in data_fetch.py'sini çalıştır:")
    print("     cd Veri_analizi && python data_fetch.py")


# ============================================================
# ÖZET TABLOSU
# ============================================================

print(f"\n{'=' * 65}")
print("  ÖZET — TÜM TÜRK UYDULARI CARA DURUMU")
print(f"{'=' * 65}")

if cara_results:
    print(f"\n  {'Uydu':<16} {'Tehdit':<22} {'Mesafe':>8} {'Pc':>12} {'CARA':>6}")
    print(f"  {'─'*16} {'─'*22} {'─'*8} {'─'*12} {'─'*6}")
    
    for name, data in sorted(cara_results.items(), 
                              key=lambda x: x[1]["cara"]["pc"], 
                              reverse=True):
        icon = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}.get(
            data["cara"]["cara_status"], "⚪"
        )
        print(f"  {name:<16} {data['threat']['name']:<22} "
              f"{data['cara']['min_distance_m']:>7.0f}m "
              f"{data['cara']['pc_scientific']:>12} "
              f"{icon} {data['cara']['cara_status']}")

print(f"\n{'=' * 65}")
print("  CANLI ENTEGRASYON TAMAMLANDI")
print(f"{'=' * 65}")
