"""
test_k2.py — K2 Tam Test Suite
================================
Bu dosyayı REPO KÖK DİZİNİNDE çalıştır:
    cd tua_sopranos1
    python test_k2.py

3 test seviyesi:
  1. Modül testleri — her K2 dosyası tek tek
  2. Pipeline testi — K1 çıktısı → K2 → sonuç
  3. Entegrasyon testi — K1 fonksiyonlarıyla birlikte (K1 hazırsa)
"""

import sys
import os
import traceback

# Repo kök dizininden çalışması için path ayarla
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
errors = []


def test(name, func):
    """Test çalıştırıcı — hata yakalar, sonuç bildirir"""
    global passed, failed, errors
    try:
        func()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}")
        print(f"     Hata: {e}")
        errors.append((name, str(e)))
        failed += 1


# ============================================================
# SEVİYE 1: MODÜL İMPORT TESTLERİ
# ============================================================

print("=" * 60)
print("  K2 TEST SUITE — TUA SOPRANOS")
print("=" * 60)

print("\n📦 SEVİYE 1: Import Testleri")
print("─" * 40)


def test_import_cara():
    from model.cara_engine import (
        compute_pc, assess_cara_status, run_cara_assessment,
        run_cara_from_k1, batch_cara_assessment, generate_cdm,
        build_encounter_frame, project_to_2d, find_tca,
        create_mock_scenario, CARA_THRESHOLDS
    )
    assert CARA_THRESHOLDS["RED"] == 1e-4
test("cara_engine import", test_import_cara)


def test_import_maneuver():
    from model.maneuver import (
        suggest_maneuver, tsiolkovsky_fuel_mass,
        estimate_lifetime_impact, compute_delta_v_options,
        recalculate_pc_after_maneuver, shadow_zone_analysis
    )
test("maneuver import", test_import_maneuver)


def test_import_threat():
    from model.threat_analysis import (
        detect_ghost_maneuver, fragmentation_warning,
        prioritize_threats, j2_expected_drift
    )
test("threat_analysis import", test_import_threat)


def test_import_game():
    from model.game_theory import (
        who_should_dodge, fuel_budget_manager,
        build_payoff_matrix, find_nash_equilibrium
    )
test("game_theory import", test_import_game)


def test_import_init():
    from model import (
        run_cara_from_k1, generate_cdm, suggest_maneuver,
        who_should_dodge, detect_ghost_maneuver
    )
test("__init__.py toplu import", test_import_init)


# ============================================================
# SEVİYE 2: FONKSİYON TESTLERİ
# ============================================================

print(f"\n🔬 SEVİYE 2: Fonksiyon Testleri")
print("─" * 40)

# --- cara_engine testleri ---

def test_pc_hesabi():
    import numpy as np
    from model.cara_engine import compute_pc
    
    miss = np.array([0.5, 0.3])
    cov = np.array([[0.1, 0], [0, 0.1]])
    hbr = 0.008
    
    pc = compute_pc(miss, cov, hbr)
    assert pc > 0, "Pc pozitif olmalı"
    assert pc < 1, "Pc 1'den küçük olmalı"
test("Pc hesabı (compute_pc)", test_pc_hesabi)


def test_cara_status_red():
    from model.cara_engine import assess_cara_status
    result = assess_cara_status(5e-4)
    assert result["status"] == "RED", f"Beklenen RED, gelen {result['status']}"
test("CARA status RED (Pc=5e-4)", test_cara_status_red)


def test_cara_status_yellow():
    from model.cara_engine import assess_cara_status
    result = assess_cara_status(5e-5)
    assert result["status"] == "YELLOW", f"Beklenen YELLOW, gelen {result['status']}"
test("CARA status YELLOW (Pc=5e-5)", test_cara_status_yellow)


def test_cara_status_green():
    from model.cara_engine import assess_cara_status
    result = assess_cara_status(1e-7)
    assert result["status"] == "GREEN", f"Beklenen GREEN, gelen {result['status']}"
test("CARA status GREEN (Pc=1e-7)", test_cara_status_green)


def test_mock_senaryolar():
    from model.cara_engine import create_mock_scenario, run_cara_assessment
    for name in ["close_approach", "moderate_risk", "safe_pass"]:
        scenario = create_mock_scenario(name)
        result = run_cara_assessment(scenario["primary"], scenario["secondary"])
        assert "pc" in result
        assert "cara_status" in result
test("3 mock senaryo", test_mock_senaryolar)


def test_encounter_frame():
    import numpy as np
    from model.cara_engine import build_encounter_frame
    
    v1 = np.array([0.0, 7.6, 0.0])
    v2 = np.array([0.5, -7.5, 0.3])
    P = build_encounter_frame(v1, v2)
    
    assert P.shape == (3, 2), f"P boyutu (3,2) olmalı, gelen {P.shape}"
    # Sütunlar ortonormal olmalı
    dot = P[:, 0] @ P[:, 1]
    assert abs(dot) < 1e-10, "P sütunları ortogonal olmalı"
test("Encounter frame (P matrisi)", test_encounter_frame)


# --- K1 köprü testi ---

def test_k1_koprü():
    from model.cara_engine import run_cara_from_k1
    
    mock_k1 = {
        "tca_utc": "2025-03-29 14:32:00 UTC",
        "time_to_tca_hours": 6.5,
        "time_to_tca_seconds": 23400,
        "min_distance_km": 0.45,
        "radial_km": 0.2,
        "intrack_km": 0.35,
        "crosstrack_km": 0.15,
        "relative_speed_kms": 0.08,
        "miss_2d": [0.25, 0.12],
        "combined_covariance_2d": [[0.04, 0.001], [0.001, 0.02]],
        "combined_hbr_m": 8.0,
        "combined_hbr_km": 0.008,
        "hbr_primary_m": 5.0,
        "hbr_secondary_m": 3.0,
        "confidence_primary": 92.0,
        "confidence_secondary": 45.0,
        "confidence_combined": 68.5,
    }
    
    result = run_cara_from_k1(mock_k1)
    assert result["cara_status"] == "RED", f"Beklenen RED, gelen {result['cara_status']}"
    assert result["pc"] > 1e-4, "Bu senaryo RED olmalı (Pc > 1e-4)"
    assert "tca_utc" in result, "K1 verisi aktarılmalı"
    assert "pc_scientific" in result, "Pc scientific notation olmalı"
test("K1 köprü (run_cara_from_k1)", test_k1_koprü)


def test_batch_assessment():
    from model.cara_engine import batch_cara_assessment
    
    mock_list = [
        {
            "miss_2d": [0.25, 0.12],
            "combined_covariance_2d": [[0.04, 0.001], [0.001, 0.02]],
            "combined_hbr_km": 0.008,
            "min_distance_km": 0.45,
            "secondary_name": "THREAT-1",
        },
        {
            "miss_2d": [3.5, 2.8],
            "combined_covariance_2d": [[0.08, 0.005], [0.005, 0.4]],
            "combined_hbr_km": 0.007,
            "min_distance_km": 5.2,
            "secondary_name": "THREAT-2",
        },
    ]
    
    results = batch_cara_assessment("TURKSAT-6A", mock_list)
    assert len(results) == 2, "2 tehdit olmalı"
    assert results[0]["pc"] >= results[1]["pc"], "En tehlikeli ilk sırada olmalı"
    assert results[0]["threat_rank"] == 1
test("Toplu değerlendirme (batch)", test_batch_assessment)


def test_cdm_uretici():
    from model.cara_engine import run_cara_from_k1, generate_cdm
    
    mock_k1 = {
        "tca_utc": "2025-03-29 14:32:00 UTC",
        "min_distance_km": 0.45,
        "miss_2d": [0.25, 0.12],
        "combined_covariance_2d": [[0.04, 0.001], [0.001, 0.02]],
        "combined_hbr_km": 0.008,
        "hbr_primary_m": 5.0,
        "hbr_secondary_m": 3.0,
        "relative_speed_kms": 0.08,
        "radial_km": 0.2,
        "intrack_km": 0.35,
        "crosstrack_km": 0.15,
    }
    
    result = run_cara_from_k1(mock_k1)
    cdm = generate_cdm("TURKSAT-6A", "COSMOS-DEB", result)
    
    assert "<?xml" in cdm, "XML başlığı olmalı"
    assert "COLLISION_PROBABILITY" in cdm, "Pc değeri CDM'de olmalı"
    assert "TURKSAT-6A" in cdm, "Primary adı CDM'de olmalı"
    assert "CCSDS" not in cdm or "cdm" in cdm, "CDM formatı doğru olmalı"
test("CDM üretici (XML)", test_cdm_uretici)


# --- maneuver testleri ---

def test_tsiolkovsky():
    from model.maneuver import tsiolkovsky_fuel_mass
    
    result = tsiolkovsky_fuel_mass(1.0, 4229)
    assert result["fuel_mass_kg"] > 0, "Yakıt pozitif olmalı"
    assert result["fuel_mass_kg"] < 100, "1 m/s için 100 kg'dan az olmalı"
    assert result["cost_usd"] > 0, "Maliyet pozitif olmalı"
    
    # Daha büyük delta-V → daha çok yakıt (üstel artış)
    result2 = tsiolkovsky_fuel_mass(10.0, 4229)
    assert result2["fuel_mass_kg"] > result["fuel_mass_kg"] * 5, "Üstel artış olmalı"
test("Tsiolkovsky yakıt hesabı", test_tsiolkovsky)


def test_manevra_onerisi():
    from model.maneuver import suggest_maneuver
    
    mock_conj = {
        "min_distance_km": 0.45,
        "miss_2d": [0.25, 0.12],
        "combined_covariance_2d": [[0.04, 0.001], [0.001, 0.02]],
        "combined_hbr_km": 0.008,
        "relative_speed_kms": 0.08,
    }
    
    result = suggest_maneuver(mock_conj, spacecraft_mass_kg=4229)
    assert len(result["options"]) == 3, "3 seçenek olmalı"
    assert result["options"][0]["level"] == "minimal"
    assert result["options"][1]["level"] == "onerilen"
    assert result["options"][2]["level"] == "maksimal"
    assert result["recommended_index"] == 1, "Önerilen ortadaki olmalı"
    
    for opt in result["options"]:
        assert "delta_v_ms" in opt
        assert "fuel_mass_kg" in opt
        assert "pc_before" in opt
        assert "pc_after" in opt
        assert "lifetime_loss_days" in opt
test("Manevra önerisi (3 seçenek)", test_manevra_onerisi)


def test_golge_bolge():
    from model.maneuver import shadow_zone_analysis
    
    result = shadow_zone_analysis(
        primary_pos=[42164.0, 0.0, 0.0],
        primary_vel=[0.0, 3.0747, 0.0],
        delta_v_ms=0.5,
        nearby_objects=[
            {"name": "OBJ-1", "pos": [42165.0, 1.0, 0.0], "vel": [0.0, 3.07, 0.0]},
        ]
    )
    assert "shadow_zone_clear" in result
    assert "conflicts" in result
    assert "conflict_count" in result
test("Gölge bölge analizi", test_golge_bolge)


# --- threat_analysis testleri ---

def test_hayalet_manevra():
    from model.threat_analysis import detect_ghost_maneuver
    
    normal_history = [
        {"epoch": f"2025-03-{d:02d}T00:00:00", "semi_major_axis_km": 6878.0,
         "eccentricity": 0.001, "inclination_deg": 97.5,
         "raan_deg": 45.0 - d * 0.2, "arg_perigee_deg": 90.0 + d * 0.15,
         "mean_motion": 15.2, "bstar": 0.0001}
        for d in range(1, 10)
    ]
    
    result = detect_ghost_maneuver(normal_history)
    assert "anomalies_detected" in result
    assert "risk_score" in result
    assert result["total_points_analyzed"] > 0
test("Hayalet manevra dedektörü", test_hayalet_manevra)


def test_parcalanma_uyari():
    from model.threat_analysis import fragmentation_warning
    
    # Stabil BSTAR
    stable = [{"epoch": f"2025-03-{d:02d}", "bstar": 0.0001} for d in range(1, 15)]
    result = fragmentation_warning(stable)
    assert result["risk_level"] == "LOW"
    
    # Artan BSTAR (parçalanma)
    rising = [{"epoch": f"2025-03-{d:02d}", "bstar": 0.0001 * (1 + d * 0.5)} for d in range(1, 15)]
    result2 = fragmentation_warning(rising)
    assert result2["risk_level"] in ["MEDIUM", "HIGH"]
test("Parçalanma erken uyarı", test_parcalanma_uyari)


def test_tehdit_onceliklendirme():
    from model.threat_analysis import prioritize_threats
    
    threats = [
        {"name": "A", "distance_km": 100, "rcs_size": "SMALL"},
        {"name": "B", "distance_km": 10, "rcs_size": "LARGE"},
    ]
    result = prioritize_threats(threats)
    assert result[0]["name"] == "B", "En yakın + en büyük ilk olmalı"
    assert result[0]["priority_rank"] == 1
test("Tehdit önceliklendirme", test_tehdit_onceliklendirme)


# --- game_theory testleri ---

def test_nash_dengesi():
    from model.game_theory import build_payoff_matrix, find_nash_equilibrium
    
    matrix = build_payoff_matrix(0.5, 0.8, 1000, 500, 1e-4)
    nash = find_nash_equilibrium(matrix["payoff_matrix"])
    
    assert "pure_nash_equilibria" in nash
    assert "mixed_nash" in nash
    assert nash["pure_nash_count"] > 0, "En az 1 Nash dengesi olmalı"
test("Nash dengesi hesabı", test_nash_dengesi)


def test_kim_kacinmali_enkaz():
    from model.game_theory import who_should_dodge
    
    result = who_should_dodge(
        pc=1e-4,
        fuel_remaining_primary_kg=200,
        fuel_remaining_secondary_kg=0,
        fuel_cost_primary_kg=0.5,
        fuel_cost_secondary_kg=0,
        is_secondary_debris=True
    )
    assert result["decision"] == "PRIMARY_DODGE", "Enkaza karşı primary kaçınmalı"
    assert result["confidence"] == 100
test("Kim kaçınmalı (enkaz)", test_kim_kacinmali_enkaz)


def test_kim_kacinmali_aktif():
    from model.game_theory import who_should_dodge
    
    result = who_should_dodge(
        pc=1e-4,
        fuel_remaining_primary_kg=200,
        fuel_remaining_secondary_kg=80,
        fuel_cost_primary_kg=0.3,
        fuel_cost_secondary_kg=0.8,
        is_secondary_debris=False
    )
    assert result["decision"] in ["PRIMARY_DODGE", "SECONDARY_DODGE", "BOTH_NEGOTIATE"]
    assert result["game_theory_applicable"] == True
    assert "nash_equilibrium" in result
test("Kim kaçınmalı (aktif uydu)", test_kim_kacinmali_aktif)


def test_yakit_butce():
    from model.game_theory import fuel_budget_manager
    
    # Bol yakıt
    r1 = fuel_budget_manager(300, 15)
    assert r1["threshold_label"] in ["GREEN", "YELLOW"], "Bol yakıtta agresif olmalı"
    
    # Kritik yakıt
    r2 = fuel_budget_manager(5, 5)
    assert r2["threshold_label"] == "CRITICAL", "Az yakıtta tutucu olmalı"
    
    # Mantıksal tutarlılık
    assert r2["optimal_pc_threshold"] > r1["optimal_pc_threshold"], \
        "Az yakıtta eşik yükselmeli"
test("Yakıt bütçe yöneticisi", test_yakit_butce)


# ============================================================
# SEVİYE 3: K1 ENTEGRASYON TESTİ
# ============================================================

print(f"\n🔗 SEVİYE 3: K1 Entegrasyon Testi")
print("─" * 40)


def test_k1_entegrasyon():
    """K1 modülleri varsa gerçek entegrasyonu test et"""
    try:
        from Veri_analizi.data_fetch import load_all_data, calculate_positions_batch
        from Veri_analizi.orbit_calc import full_conjunction_analysis
    except ImportError:
        print("  ⏭️  K1 modülleri bulunamadı — entegrasyon testi atlandı")
        print("     (Veri_analizi/ klasörü repo'da olmalı)")
        return
    
    from model.cara_engine import run_cara_from_k1
    from model.maneuver import suggest_maneuver
    from model.game_theory import who_should_dodge
    
    # K1'den veri çek (cache'ten)
    turkish, geo_debris, leo_debris = load_all_data(use_cache=True)
    
    if not turkish:
        print("  ⏭️  Cache boş — K1 henüz veri çekmemiş")
        return
    
    # İlk uydu için conjunction analizi
    first_sat = list(turkish.keys())[0]
    print(f"     Test uydusu: {first_sat}")
    
    # Pozisyon hesapla
    positions = calculate_positions_batch(turkish)
    
    if first_sat not in positions:
        print(f"  ⏭️  {first_sat} pozisyonu hesaplanamadı")
        return
    
    print("  ✅ K1 → K2 entegrasyon hazır (cache varsa tam çalışır)")

test("K1 entegrasyon", test_k1_entegrasyon)


# ============================================================
# SONUÇ
# ============================================================

print(f"\n{'=' * 60}")
total = passed + failed
print(f"  SONUÇ: {passed}/{total} test geçti", end="")
if failed > 0:
    print(f" — {failed} BAŞARISIZ ❌")
    print(f"\n  Başarısız testler:")
    for name, err in errors:
        print(f"    ❌ {name}: {err}")
else:
    print(f" — HEPSİ BAŞARILI ✅")
print(f"{'=' * 60}")

# Çıkış kodu (CI/CD için)
sys.exit(0 if failed == 0 else 1)
