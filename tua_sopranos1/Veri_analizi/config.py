# ============================================
# YÖRÜNGE KALKANI — CONFIG
# ============================================

# Space-Track API
SPACETRACK_USER = "ozaninaltekin@gmail.com"
SPACETRACK_PASS = "OzanInaltekin01.."

# Türkiye'nin Tüm Uydu Filosu
TURKISH_SATELLITES = {
    # === AKTİF HABERLEŞME (GEO) ===
    "Turksat 3A":  {"norad_id": 33056, "orbit": "GEO", "type": "COMMS", "status": "active", "mass_kg": 3117, "launch_year": 2008},
    "Turksat 4A":  {"norad_id": 39522, "orbit": "GEO", "type": "COMMS", "status": "active", "mass_kg": 4869, "launch_year": 2014},
    "Turksat 4B":  {"norad_id": 40984, "orbit": "GEO", "type": "COMMS", "status": "active", "mass_kg": 4860, "launch_year": 2015},
    "Turksat 5A":  {"norad_id": 47306, "orbit": "GEO", "type": "COMMS", "status": "active", "mass_kg": 3500, "launch_year": 2021},
    "Turksat 5B":  {"norad_id": 50212, "orbit": "GEO", "type": "COMMS", "status": "active", "mass_kg": 4500, "launch_year": 2021},
    "Turksat 6A":  {"norad_id": 60233, "orbit": "GEO", "type": "COMMS", "status": "active", "mass_kg": 4229, "launch_year": 2024},

    # === AKTİF GÖZLEM/KEŞİF (LEO) ===
    "Gokturk-1":   {"norad_id": 41875, "orbit": "LEO", "type": "RECON", "status": "active", "mass_kg": 1060, "launch_year": 2016},
    "Gokturk-2":   {"norad_id": 39030, "orbit": "LEO", "type": "RECON", "status": "active", "mass_kg": 409,  "launch_year": 2012},
    "IMECE":        {"norad_id": 56197, "orbit": "LEO", "type": "EO",    "status": "active", "mass_kg": 700,  "launch_year": 2023},

    # === KÜP UYDU (LEO) ===
    "Turksat 3U":   {"norad_id": 39152, "orbit": "LEO", "type": "CUBE",  "status": "active", "mass_kg": 3,    "launch_year": 2013},

    # === EMEKLİ (kendileri artık birer uzay çöpü) ===
    "Turksat 1B":   {"norad_id": 23200, "orbit": "GEO", "type": "COMMS", "status": "retired", "mass_kg": 1530, "launch_year": 1994},
    "Turksat 1C":   {"norad_id": 23949, "orbit": "GEO", "type": "COMMS", "status": "retired", "mass_kg": 1530, "launch_year": 1996},
    "Turksat 2A":   {"norad_id": 26666, "orbit": "GEO", "type": "COMMS", "status": "retired", "mass_kg": 3530, "launch_year": 2001},
}

# CARA Eşik Değerleri (NASA standardı)
CARA_THRESHOLDS = {
    "RED":    1e-4,   # Pc > 1e-4 → manevra gerekli
    "YELLOW": 1e-5,   # 1e-5 < Pc < 1e-4 → izle
    "GREEN":  1e-5,   # Pc < 1e-5 → güvenli
}

# Çöp Tarama Filtreleri
DEBRIS_FILTERS = {
    "GEO": {"mean_motion_min": 0.95, "mean_motion_max": 1.05, "limit": 500},
    "LEO": {"mean_motion_min": 14.0, "mean_motion_max": 16.0, "limit": 500},
}

# Sentetik Kovaryans Parametreleri (CARA uyumlu)
COVARIANCE_PARAMS = {
    "LEO": {"pos_sigma_per_hour": 0.5},   # km/saat belirsizlik artışı
    "GEO": {"pos_sigma_per_hour": 0.1},
}

# Fiziksel Sabitler
EARTH_RADIUS_KM = 6371.0
GEO_ALTITUDE_KM = 35786.0

# Uydu bazlı operasyonel parametreler
# Yakıt tahminleri: fırlatma yılı + tip + kütleden hesaplanmış
# mission_years_remaining: günümüzden itibaren beklenen görev süresi
SATELLITE_OPERATIONS = {
    "Turksat 3A":  {"fuel_kg": 80,   "mission_years_remaining": 3},
    "Turksat 4A":  {"fuel_kg": 150,  "mission_years_remaining": 8},
    "Turksat 4B":  {"fuel_kg": 160,  "mission_years_remaining": 9},
    "Turksat 5A":  {"fuel_kg": 280,  "mission_years_remaining": 11},
    "Turksat 5B":  {"fuel_kg": 300,  "mission_years_remaining": 11},
    "Turksat 6A":  {"fuel_kg": 350,  "mission_years_remaining": 14},
    "Gokturk-1":   {"fuel_kg": 50,   "mission_years_remaining": 3},
    "Gokturk-2":   {"fuel_kg": 30,   "mission_years_remaining": 2},
    "IMECE":       {"fuel_kg": 60,   "mission_years_remaining": 3},
    "Turksat 3U":  {"fuel_kg": 0.5,  "mission_years_remaining": 1},
    "Turksat 1B":  {"fuel_kg": 0,    "mission_years_remaining": 0},  # emekli
    "Turksat 1C":  {"fuel_kg": 0,    "mission_years_remaining": 0},  # emekli
    "Turksat 2A":  {"fuel_kg": 0,    "mission_years_remaining": 0},  # emekli
}