"""
maneuver.py — Manevra Karar ve Maliyet Motoru
===============================================
Çarpışma riski tespit edildiğinde 3 farklı manevra seçeneği sunar.
Her seçenek için yakıt maliyeti, ömür kaybı ve yeni Pc hesaplar.

Temel Fizik:
  Bir uydu yörünge değiştirmek için iticilerini ateşler.
  Bu "delta-V" (hız değişimi) denen bir büyüklük üretir.
  Ne kadar çok delta-V → o kadar çok yakıt harcanır.
  Tsiolkovsky denklemi bu ilişkiyi tanımlar.

Referanslar:
  - Tsiolkovsky Rocket Equation (1903)
  - Vallado, D.A. "Fundamentals of Astrodynamics and Applications"
  - NASA CARA Maneuver Planning Guidelines

Yazar: K2 Algoritma Mühendisi
Proje: TUA SOPRANOS
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from .cara_engine import compute_pc, assess_cara_status, build_encounter_frame, project_to_2d
except ImportError:
    from cara_engine import compute_pc, assess_cara_status, build_encounter_frame, project_to_2d


# ============================================================
# BÖLÜM 1: FİZİKSEL SABİTLER
# ============================================================

# Yerçekimi sabiti × Dünya kütlesi (km³/s²)
MU_EARTH = 398600.4418

# Tipik uydu itici verimliliği — özgül impuls (saniye)
# Bipropellant (hidrazin) iticiler için tipik değer
TYPICAL_ISP = 320  # saniye

# Yerçekimi ivmesi (m/s²) — Tsiolkovsky'de kullanılır
G0 = 9.80665  # m/s²

# Yakıt maliyeti (USD/kg — fırlatma maliyeti dahil)
FUEL_COST_PER_KG = 50000  # ~50k USD/kg yörüngede yakıt

# Yıllık beklenen conjunction sayısı (istatistiksel ortalama)
ANNUAL_CONJUNCTIONS = {
    "LEO": 15,   # LEO'da yılda ~15 ciddi yaklaşma
    "GEO": 5,    # GEO'da yılda ~5 ciddi yaklaşma
}


# ============================================================
# BÖLÜM 2: TSİOLKOVSKY ROKET DENKLEMİ
# ============================================================

def tsiolkovsky_fuel_mass(
    delta_v_ms: float,    # hız değişimi (m/s)
    spacecraft_mass_kg: float,  # uydu kuru kütlesi (kg)
    isp: float = TYPICAL_ISP    # özgül impuls (s)
) -> dict:
    """
    Tsiolkovsky Roket Denklemi ile yakıt kütlesi hesabı.
    
    Denklem:
      Δv = Isp × g₀ × ln(m_initial / m_final)
    
    Burada:
      Δv     = hız değişimi (m/s)
      Isp    = özgül impuls — iticinin verimliliği (s)
      g₀     = yerçekimi ivmesi (9.80665 m/s²)
      m_initial = başlangıç kütlesi (uydu + yakıt)
      m_final   = bitiş kütlesi (uydu, yakıt harcanmış)
    
    Çözüm (yakıt kütlesi):
      m_fuel = m_final × (exp(Δv / (Isp × g₀)) - 1)
    
    Sezgisel:
      Δv arttıkça yakıt ÜSTEL olarak artar — doğrusal değil!
      Bu yüzden her manevra çok değerlidir. Boşa yakıt yakılmamalı.
    
    Jüriye söyle:
      "Tsiolkovsky denklemi 1903'ten beri roket biliminin temel
       denklemidir. Yakıt tüketiminin hız değişimiyle üstel ilişkisini
       tanımlar — bu yüzden her manevra kararı kritik."
    """
    
    # Efektif egzoz hızı
    v_exhaust = isp * G0  # m/s
    
    # Kütle oranı: exp(Δv / v_exhaust)
    mass_ratio = np.exp(delta_v_ms / v_exhaust)
    
    # Yakıt kütlesi
    fuel_mass = spacecraft_mass_kg * (mass_ratio - 1)
    
    # Yakıt yüzdesi (toplam kütleye göre)
    fuel_percent = (fuel_mass / (spacecraft_mass_kg + fuel_mass)) * 100
    
    return {
        "delta_v_ms": round(delta_v_ms, 4),
        "fuel_mass_kg": round(fuel_mass, 4),
        "fuel_percent": round(fuel_percent, 2),
        "mass_ratio": round(mass_ratio, 6),
        "exhaust_velocity_ms": round(v_exhaust, 1),
        "cost_usd": round(fuel_mass * FUEL_COST_PER_KG, 2),
    }


# ============================================================
# BÖLÜM 3: ÖMÜR ETKİSİ HESABI
# ============================================================

def estimate_lifetime_impact(
    fuel_mass_kg: float,
    total_fuel_remaining_kg: float,
    mission_remaining_years: float,
    orbit_type: str = "GEO"
) -> dict:
    """
    Manevranın uydu ömrüne etkisini hesaplar.
    
    Mantık:
      Uydunun kalan yakıtı sınırlı. Her manevra bu yakıttan yer.
      Yakıt bitince uydu kontrol edilemez → görev sonu.
      
      Basit model:
        Kalan yakıt oransal olarak ömrü belirler.
        fuel_used / total_fuel × remaining_years = kaybedilen ömür
    
    Gerçekte daha karmaşık (istasyon tutma, yörünge düzeltmeleri)
    ama sunum için bu yaklaşım yeterli ve anlaşılır.
    """
    
    if total_fuel_remaining_kg <= 0:
        return {
            "lifetime_loss_years": 0,
            "lifetime_loss_days": 0,
            "lifetime_remaining_years": 0,
            "fuel_remaining_after_kg": 0,
            "fuel_remaining_percent": 0,
            "warning": "Yakıt verisi yok"
        }
    
    # Yakıt oranı
    fuel_ratio = fuel_mass_kg / total_fuel_remaining_kg
    
    # Ömür kaybı
    lifetime_loss_years = fuel_ratio * mission_remaining_years
    
    # Kalan yakıt
    fuel_after = total_fuel_remaining_kg - fuel_mass_kg
    fuel_remaining_pct = (fuel_after / total_fuel_remaining_kg) * 100
    
    return {
        "lifetime_loss_years": round(lifetime_loss_years, 3),
        "lifetime_loss_days": round(lifetime_loss_years * 365, 1),
        "lifetime_remaining_years": round(mission_remaining_years - lifetime_loss_years, 2),
        "fuel_remaining_after_kg": round(max(0, fuel_after), 2),
        "fuel_remaining_percent": round(max(0, fuel_remaining_pct), 1),
    }


# ============================================================
# BÖLÜM 4: DELTA-V HESABI (Manevra Büyüklüğü)
# ============================================================

def compute_delta_v_options(
    miss_distance_km: float,
    hbr_combined_km: float,
    relative_speed_kms: float,
    orbit_type: str = "GEO"
) -> list:
    """
    3 farklı manevra seçeneği üretir: minimal, önerilen, maksimal.
    
    Fizik:
      Manevranın amacı miss distance'ı artırmak.
      In-track yönde (yörünge boyunca) yapılan manevra en etkilidir
      çünkü uydu zamanla daha uzağa kayar.
      
      Yaklaşık ilişki:
        Δ(miss_distance) ≈ Δv × T_warning / 2
      
      T_warning = TCA'ya kalan süre. Erken manevra → küçük Δv yeterli.
    
    3 Seçenek:
      1. MİNİMAL:  miss distance'ı HBR'ın 5 katına çıkar
         → "Kıl payı kurtarırız, yakıt minimum"
      2. ÖNERİLEN: miss distance'ı HBR'ın 20 katına çıkar
         → "Güvenli mesafe, makul yakıt" (CARA standardı)
      3. MAKSİMAL: miss distance'ı HBR'ın 50 katına çıkar
         → "Kesin güvenli, yakıt pahalı"
    """
    
    # Hedef miss distance çarpanları (HBR katı)
    multipliers = {
        "minimal":   5,    # HBR × 5
        "onerilen": 20,    # HBR × 20 (CARA önerisi)
        "maksimal": 50,    # HBR × 50
    }
    
    options = []
    
    for level, mult in multipliers.items():
        # Hedef miss distance
        target_miss_km = hbr_combined_km * mult
        
        # Gerekli yer değiştirme (mevcut miss distance'ı çıkar)
        needed_shift_km = max(0, target_miss_km - miss_distance_km)
        
        # Delta-V yaklaşık hesabı
        # GEO'da küçük Δv yeterli (yavaş drift), LEO'da daha büyük gerekir
        if orbit_type == "GEO":
            # GEO: Hohmann transfer yaklaşımı
            # Δv ≈ needed_shift / (orbital_period/4)
            # GEO orbital period ≈ 86400 s, /4 ≈ 21600 s
            delta_v_kms = needed_shift_km / 21600  # çok kaba yaklaşım
        else:
            # LEO: daha yüksek Δv gerekir
            # Δv ≈ needed_shift / (orbital_period/4)
            # LEO orbital period ≈ 5400 s, /4 ≈ 1350 s
            delta_v_kms = needed_shift_km / 1350
        
        delta_v_ms = delta_v_kms * 1000  # m/s'ye dönüştür
        
        # Minimum Δv: en az 0.01 m/s (pratik alt sınır)
        delta_v_ms = max(delta_v_ms, 0.01)
        
        options.append({
            "level": level,
            "level_tr": {
                "minimal": "Minimal (Kıl payı)",
                "onerilen": "Önerilen (CARA standardı)",
                "maksimal": "Maksimal (Kesin güvenli)"
            }[level],
            "delta_v_ms": round(delta_v_ms, 4),
            "target_miss_km": round(target_miss_km, 4),
            "target_miss_m": round(target_miss_km * 1000, 1),
            "hbr_multiplier": mult,
        })
    
    return options


# ============================================================
# BÖLÜM 5: MANEVRA SONRASI Pc YENİDEN HESAPLAMA
# ============================================================

def recalculate_pc_after_maneuver(
    original_miss_2d: list,
    cov_2d: list,
    hbr_km: float,
    delta_v_ms: float,
    maneuver_direction: str = "intrack"
) -> dict:
    """
    Manevra sonrası yeni Pc hesaplar.
    
    Mantık:
      Manevra, miss distance vektörünü değiştirir.
      In-track manevra → miss_2d'nin x bileşenini artırır (basitleştirme)
      Yeni miss_2d ile Pc'yi tekrar hesapla.
      
    Bu fonksiyon K3'ün manevra panelinde "Manevra Uygula" butonuna
    basıldığında çağrılır. "Bu manevra Pc'yi şu kadar düşürür" gösterir.
    """
    
    miss_2d = np.array(original_miss_2d)
    cov = np.array(cov_2d)
    
    # Manevranın miss distance'a etkisi (basitleştirilmiş)
    # Gerçekte propagation gerekir, biz yaklaşık hesaplıyoruz
    delta_v_km = delta_v_ms / 1000  # km/s'ye dönüştür
    
    # In-track manevra: miss_2d'nin birinci bileşenine ekle
    # (çarpışma düzlemindeki ilk eksen genelde in-track'e yakın)
    shift = delta_v_km * 100  # basitleştirilmiş çarpan
    
    new_miss_2d = miss_2d.copy()
    if maneuver_direction == "intrack":
        new_miss_2d[0] += shift
    elif maneuver_direction == "crosstrack":
        new_miss_2d[1] += shift
    else:  # radial
        new_miss_2d[0] += shift * 0.5
        new_miss_2d[1] += shift * 0.5
    
    # Yeni Pc
    new_pc = compute_pc(new_miss_2d, cov, hbr_km)
    new_cara = assess_cara_status(new_pc)
    
    # Orijinal Pc
    original_pc = compute_pc(miss_2d, cov, hbr_km)
    
    # İyileşme oranı
    if original_pc > 0:
        improvement = original_pc / max(new_pc, 1e-30)
    else:
        improvement = 0
    
    return {
        "original_pc": original_pc,
        "original_pc_scientific": f"{original_pc:.2e}",
        "new_pc": new_pc,
        "new_pc_scientific": f"{new_pc:.2e}",
        "new_cara_status": new_cara["status"],
        "improvement_factor": round(improvement, 1),
        "improvement_text": f"Pc {improvement:.0f}x düştü" if improvement > 1 else "Değişim yok",
        "new_miss_2d_km": new_miss_2d.tolist(),
    }


# ============================================================
# BÖLÜM 6: TAM MANEVRA ÖNERİSİ (ANA FONKSİYON)
# ============================================================

def suggest_maneuver(
    conjunction_data: dict,
    spacecraft_mass_kg: float = 4229,    # Türksat 6A varsayılan
    fuel_remaining_kg: float = 200,       # tahmini kalan yakıt
    mission_remaining_years: float = 12,  # kalan görev süresi
    orbit_type: str = "GEO",
    isp: float = TYPICAL_ISP
) -> dict:
    """
    TAM MANEVRA ÖNERİ PAKETİ — K3'ün çağıracağı ana fonksiyon.
    
    Girdi: K1'in full_conjunction_analysis() çıktısı + uydu parametreleri
    Çıktı: 3 manevra seçeneği, her biri için:
      - Delta-V (m/s)
      - Yakıt maliyeti (kg ve USD)
      - Ömür kaybı (gün)
      - Manevra sonrası yeni Pc
      - CARA durumu değişimi
    
    Bu fonksiyon K3'ün manevra panelinde 3 kart olarak gösterilir.
    """
    
    miss_distance_km = conjunction_data.get("min_distance_km", 1.0)
    miss_2d = conjunction_data.get("miss_2d", [0.5, 0.3])
    cov_2d = conjunction_data.get("combined_covariance_2d", [[0.1, 0], [0, 0.1]])
    hbr_km = conjunction_data.get("combined_hbr_km", 0.008)
    rel_speed = conjunction_data.get("relative_speed_kms", 1.0)
    
    # 3 delta-V seçeneği hesapla
    dv_options = compute_delta_v_options(
        miss_distance_km, hbr_km, rel_speed, orbit_type
    )
    
    results = []
    
    for opt in dv_options:
        # Yakıt hesabı (Tsiolkovsky)
        fuel = tsiolkovsky_fuel_mass(
            opt["delta_v_ms"], spacecraft_mass_kg, isp
        )
        
        # Ömür etkisi
        lifetime = estimate_lifetime_impact(
            fuel["fuel_mass_kg"],
            fuel_remaining_kg,
            mission_remaining_years,
            orbit_type
        )
        
        # Manevra sonrası Pc
        pc_after = recalculate_pc_after_maneuver(
            miss_2d, cov_2d, hbr_km, opt["delta_v_ms"]
        )
        
        results.append({
            # Manevra bilgileri
            "level": opt["level"],
            "level_tr": opt["level_tr"],
            "delta_v_ms": opt["delta_v_ms"],
            "target_miss_m": opt["target_miss_m"],
            "hbr_multiplier": opt["hbr_multiplier"],
            
            # Yakıt maliyeti
            "fuel_mass_kg": fuel["fuel_mass_kg"],
            "fuel_cost_usd": fuel["cost_usd"],
            "fuel_percent_of_remaining": round(
                (fuel["fuel_mass_kg"] / max(fuel_remaining_kg, 0.01)) * 100, 2
            ),
            
            # Ömür etkisi
            "lifetime_loss_days": lifetime["lifetime_loss_days"],
            "lifetime_remaining_years": lifetime["lifetime_remaining_years"],
            "fuel_remaining_after_kg": lifetime["fuel_remaining_after_kg"],
            
            # Pc değişimi
            "pc_before": pc_after["original_pc_scientific"],
            "pc_after": pc_after["new_pc_scientific"],
            "new_cara_status": pc_after["new_cara_status"],
            "improvement_factor": pc_after["improvement_factor"],
            "improvement_text": pc_after["improvement_text"],
        })
    
    # Öneri: ortadaki seçenek (önerilen)
    recommended_idx = 1
    
    return {
        "options": results,
        "recommended": results[recommended_idx],
        "recommended_index": recommended_idx,
        "spacecraft_mass_kg": spacecraft_mass_kg,
        "fuel_remaining_kg": fuel_remaining_kg,
        "orbit_type": orbit_type,
    }


# ============================================================
# BÖLÜM 7: GÖLGE BÖLGE ANALİZİ (#20)
# ============================================================

def shadow_zone_analysis(
    primary_pos: list,
    primary_vel: list,
    delta_v_ms: float,
    nearby_objects: list,
    hbr_primary: float = 0.005
) -> dict:
    """
    Manevra Gölge Bölgesi Analizi (#20)
    
    Problem:
      Bir tehditten kaçınmak için manevra yapıyorsun.
      Ama yeni yörünge başka bir nesneye yaklaştırabilir!
      Buna "gölge bölge" (shadow zone) denir.
    
    Yöntem:
      1. Manevra sonrası yeni pozisyonu tahmin et
      2. Çevredeki tüm nesnelerle mesafe hesapla
      3. Herhangi birinin miss distance'ı kötüleştiyse UYARI ver
    
    Args:
        primary_pos: Uydu pozisyonu [x,y,z] km
        primary_vel: Uydu hızı [vx,vy,vz] km/s
        delta_v_ms: Manevra büyüklüğü m/s
        nearby_objects: Çevredeki nesneler listesi
            [{"name": str, "pos": [x,y,z], "vel": [vx,vy,vz]}, ...]
        hbr_primary: Primary HBR km
    """
    
    pos = np.array(primary_pos)
    vel = np.array(primary_vel)
    
    # Manevra yönü: in-track (hız yönünde)
    vel_unit = vel / np.linalg.norm(vel)
    delta_v_km = delta_v_ms / 1000
    new_vel = vel + vel_unit * delta_v_km
    
    # Basitleştirilmiş propagation: 1 saat sonraki pozisyon
    dt = 3600  # saniye
    new_pos = pos + new_vel * (dt / 1.0)  # km (çok kaba yaklaşım)
    old_future_pos = pos + vel * (dt / 1.0)
    
    conflicts = []
    safe = []
    
    for obj in nearby_objects:
        obj_pos = np.array(obj["pos"])
        
        # Eski mesafe
        old_dist = np.linalg.norm(old_future_pos - obj_pos)
        
        # Yeni mesafe (manevra sonrası)
        new_dist = np.linalg.norm(new_pos - obj_pos)
        
        # Mesafe kötüleşti mi?
        dist_change = new_dist - old_dist
        
        entry = {
            "name": obj.get("name", "UNKNOWN"),
            "old_distance_km": round(old_dist, 2),
            "new_distance_km": round(new_dist, 2),
            "change_km": round(dist_change, 2),
            "worsened": dist_change < 0,
        }
        
        if dist_change < -1.0:  # 1 km'den fazla kötüleşme
            entry["status"] = "WARNING"
            conflicts.append(entry)
        else:
            entry["status"] = "SAFE"
            safe.append(entry)
    
    return {
        "total_objects_checked": len(nearby_objects),
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "safe_count": len(safe),
        "shadow_zone_clear": len(conflicts) == 0,
        "recommendation": "Gölge bölge temiz — manevra güvenli." 
                          if len(conflicts) == 0 
                          else f"UYARI: {len(conflicts)} nesne ile yeni risk oluşabilir!"
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  MANEVRA MOTORU — TEST SUITE")
    print("  TUA SOPRANOS — K2")
    print("=" * 60)
    
    # K1 çıktısını simüle et (RED senaryo)
    mock_conjunction = {
        "min_distance_km": 0.45,
        "miss_2d": [0.25, 0.12],
        "combined_covariance_2d": [[0.04, 0.001], [0.001, 0.02]],
        "combined_hbr_km": 0.008,
        "combined_hbr_m": 8.0,
        "relative_speed_kms": 0.08,
        "primary_pos_tca": [42164.0, 0.0, 0.0],
        "primary_vel_tca": [0.0, 3.0747, 0.0],
    }
    
    # Manevra önerisi
    print("\n📋 TURKSAT-6A Manevra Önerisi")
    print("─" * 50)
    
    result = suggest_maneuver(
        mock_conjunction,
        spacecraft_mass_kg=4229,   # Türksat 6A
        fuel_remaining_kg=200,
        mission_remaining_years=12,
        orbit_type="GEO"
    )
    
    for i, opt in enumerate(result["options"]):
        marker = " ★ ÖNERİLEN" if i == result["recommended_index"] else ""
        print(f"\n  Seçenek {i+1}: {opt['level_tr']}{marker}")
        print(f"    Delta-V:         {opt['delta_v_ms']:.4f} m/s")
        print(f"    Hedef Mesafe:    {opt['target_miss_m']} m ({opt['hbr_multiplier']}×HBR)")
        print(f"    Yakıt:           {opt['fuel_mass_kg']:.4f} kg")
        print(f"    Yakıt Maliyeti:  ${opt['fuel_cost_usd']:,.2f}")
        print(f"    Yakıt Oranı:     %{opt['fuel_percent_of_remaining']}")
        print(f"    Ömür Kaybı:      {opt['lifetime_loss_days']} gün")
        print(f"    Pc Önce:         {opt['pc_before']}")
        print(f"    Pc Sonra:        {opt['pc_after']}")
        print(f"    CARA Durumu:     {opt['new_cara_status']}")
        print(f"    İyileşme:        {opt['improvement_text']}")
    
    # Tsiolkovsky testi
    print(f"\n{'─' * 50}")
    print("  Tsiolkovsky Detay Testi")
    print(f"{'─' * 50}")
    
    for dv in [0.1, 1.0, 5.0, 10.0]:
        fuel = tsiolkovsky_fuel_mass(dv, 4229)
        print(f"  Δv={dv:>5.1f} m/s → yakıt={fuel['fuel_mass_kg']:.3f} kg"
              f"  maliyet=${fuel['cost_usd']:>12,.2f}")
    
    # Gölge bölge testi
    print(f"\n{'─' * 50}")
    print("  Gölge Bölge Analizi")
    print(f"{'─' * 50}")
    
    shadow = shadow_zone_analysis(
        primary_pos=[42164.0, 0.0, 0.0],
        primary_vel=[0.0, 3.0747, 0.0],
        delta_v_ms=0.5,
        nearby_objects=[
            {"name": "ASTRA-1M", "pos": [42165.0, 1.0, 0.0], "vel": [0.0, 3.07, 0.0]},
            {"name": "SES-5", "pos": [42170.0, 5.0, 0.0], "vel": [0.0, 3.08, 0.0]},
            {"name": "COSMOS-DEB", "pos": [42164.5, 0.3, 0.1], "vel": [0.1, 3.06, 0.0]},
        ]
    )
    
    print(f"  Kontrol edilen: {shadow['total_objects_checked']} nesne")
    print(f"  Çakışma:        {shadow['conflict_count']}")
    print(f"  Güvenli:        {shadow['safe_count']}")
    print(f"  Sonuç:          {shadow['recommendation']}")
    
    print(f"\n{'=' * 60}")
    print("  MANEVRA MOTORU TESTLERİ TAMAMLANDI")
    print(f"{'=' * 60}")
