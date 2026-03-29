# TUA SOPRANOS
### Türk Uydu Güvenlik ve Çarpışma Önleme Sistemi

Türkiye'nin yörüngede bulunan uydularını çöp ve diğer uzay nesnelerinden korumak için geliştirilmiş yapay zeka destekli bir çarpışma riski analiz sistemidir.

---

## Proje Özeti

TUA SOPRANOS, gerçek zamanlı TLE verilerini Space-Track API üzerinden çekerek SGP4 yörünge propagasyonu, NASA CARA Pc hesabı, XGBoost risk sınıflandırması ve LSTM yörünge düzeltmesiyle Türk uydularını sürekli izler. Tehdit tespit edildiğinde oyun teorisi tabanlı manevra kararı üretir ve CCSDS 508.0 uyumlu CDM dosyası oluşturur.

---

## İzlenen Uydular

| Uydu | NORAD | Yörünge | Tip | Kütle |
|------|-------|---------|-----|-------|
| Turksat 3A | 33056 | GEO | Haberleşme | 3117 kg |
| Turksat 4A | 39522 | GEO | Haberleşme | 4869 kg |
| Turksat 4B | 40984 | GEO | Haberleşme | 4860 kg |
| Turksat 5A | 47306 | GEO | Haberleşme | 3500 kg |
| Turksat 5B | 50212 | GEO | Haberleşme | 4500 kg |
| Turksat 6A | 60233 | GEO | Haberleşme | 4229 kg |
| Göktürk-1 | 41875 | LEO | Keşif | 1060 kg |
| Göktürk-2 | 39030 | LEO | Keşif | 409 kg |
| IMECE | 56197 | LEO | Gözlem | 700 kg |
| Turksat 3U | 39152 | LEO | CubeSat | 3 kg |

---

## Sistem Mimarisi

```
Space-Track API
      │
      ▼
┌─────────────┐
│     K1      │  Veri_analizi/
│  Veri Katmanı│  ├── config.py        → Uydu listesi, CARA eşikleri
│             │  ├── data_fetch.py     → TLE çekimi, SGP4, tehdit tarama
│             │  ├── orbit_calc.py     → RIC koordinat, kovaryans projeksiyonu
│             │  └── threat_analysis.py→ Önceliklendirme skoru
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     K2      │  model/
│  Model Katmanı│  ├── ml_training.py   → XGBoost + LSTM eğitim pipeline
│             │  ├── ml_model.py       → Çıkarım, hybrid pipeline
│             │  ├── cara_engine.py    → NASA CARA 2D Pc hesabı (dblquad)
│             │  ├── game_theory.py    → Nash dengesi, yakıt bütçesi
│             │  ├── maneuver.py       → Delta-V hesabı, manevra seçenekleri
│             │  ├── threat_analysis.py→ K2 tehdit skorlama
│             │  └── model_evaluation.py→ Confusion matrix, ROC-AUC raporu
└─────────────┘
       │
       ▼
  run_k2_live.py   → Uçtan uca canlı entegrasyon
  test_k2.py       → 41 testlik doğrulama paketi
```

---

## Kurulum

### Gereksinimler

```bash
pip install sgp4 xgboost scikit-learn numpy scipy requests torch
```

### Space-Track Hesabı

`tua_sopranos1/Veri_analizi/config.py` dosyasına kendi bilgilerinizi girin:

```python
SPACETRACK_USER = "email@example.com"
SPACETRACK_PASS = "sifreniz"
```

---

## Kullanım

### Canlı Analiz

```bash
cd tua_sopranos1
python run_k2_live.py
```

Çıktı adımları:
1. **Veri Yükleme** — Space-Track'ten TLE çekimi (cache varsa cache'ten)
2. **Pozisyon Hesabı** — SGP4 ile anlık pozisyon + güven skoru
3. **Tehdit Tarama** — Her uydu için en yakın 5 tehdit
4. **CARA Pc Hesabı** — TCA propagasyonu + 2D çarpışma olasılığı
5. **Manevra Önerisi** — 3 seçenekli Delta-V hesabı
6. **Oyun Teorisi** — Nash dengesi ile manevra kararı
7. **Yakıt Bütçesi** — Optimal Pc eşiği belirleme
8. **CDM Üretimi** — CCSDS 508.0 uyumlu XML dosyası

### Testler

```bash
cd tua_sopranos1
python test_k2.py
```

41 test — 4 seviye:
- Level 1: Import kontrolleri
- Level 2: Fizik fonksiyonları (SGP4, Pc, RIC)
- Level 3: ML model çıkarımı
- Level 4: Uçtan uca entegrasyon

---

## ML Modelleri

### XGBoost Risk Sınıflandırıcı

18 özellik üzerine eğitilmiş çok sınıflı sınıflandırıcı:

| Özellik | Açıklama |
|---------|----------|
| miss_km | 2D kaçırma mesafesi |
| mahalanobis | Miss/sigma oranı |
| rel_vel | Göreli hız (km/s) |
| hbr_m | Hard Body Radius (m) |
| altitude | Yörünge yüksekliği |
| cov_trace | Kovaryans izi |
| tca_hours | TCA'ya kalan süre |
| ... | (18 özellik toplam) |

Çıktı: `GREEN` / `YELLOW` / `RED`

| Sınıf | Pc Aralığı | Aksiyon |
|-------|-----------|---------|
| GREEN | Pc ≤ 1e-5 | Rutin izleme |
| YELLOW | 1e-5 < Pc ≤ 1e-4 | İzle, plan hazırla |
| RED | Pc > 1e-4 | Acil manevra |

### LSTM Yörünge Düzeltici

SGP4'ün artık hatalarını öğrenen 2 katmanlı LSTM:
- Giriş: 48 saatlik SGP4 yörüngesi `(batch, 48, 7)`
- Çıkış: 24 saatlik düzeltme residualı `(batch, 24, 3)`

### Eğitim Verisi

Üç kaynaktan oluşan karma dataset:
1. **Zaman Serisi Tarama** — 7 gün / 1 saat adım, 13 uydu × 1000 çöp
2. **Geçmiş TLE Analizi** — Son 30 günün pozisyon geçmişi
3. **SOCRATES CSV** — CelesTrak conjunction veritabanı (131.677 kayıt)

### Modeli Yeniden Eğitmek (Google Colab — GPU)

```
tua_sopranos1/TUA_SOPRANOS_Colab_Training.ipynb
```

Eğitim sonrası indirilen dosyaları `tua_sopranos1/model/` klasörüne koy:

```
model/
├── xgboost_risk_model.pkl
├── lstm_orbit_model.pt
├── lstm_scaler.pkl
└── evaluation_report.json
```

---

## Fizik Metodolojisi

### CARA 2D Pc Hesabı

NASA CARA (Conjunction Assessment Risk Analysis) standardı:

```
Pc = ∬_{x²+y² ≤ HBR²} f(x,y) dx dy

f(x,y) = bivariate normal PDF
HBR    = Hard Body Radius (iki nesnenin fiziksel boyutlarının toplamı)
```

`miss_km` yüzlerce km olduğunda Pc matematiksel olarak sıfıra yaklaşır — bu fiziksel olarak doğrudur.

### SGP4 Güven Skoru

```
Güven = 100% - (TLE_yaş_saat × 1.6%)
LEO: atmosfer direnci nedeniyle güven daha hızlı düşer
GEO: saatte ~0.14% bozunma
```

### Nash Dengesi Manevra Kararı

İki aktörlü oyun teorisi: Türk uydusu vs. tehdit nesnesi. Yakıt oranı, nesne tipi (debris/aktif) ve Pc değerine göre optimal strateji belirlenir.

---

## Proje Yapısı

```
TUA_SOPRANOS/
└── tua_sopranos1/
    ├── Veri_analizi/
    │   ├── config.py
    │   ├── data_fetch.py
    │   ├── orbit_calc.py
    │   └── threat_analysis.py
    ├── model/
    │   ├── ml_training.py
    │   ├── ml_model.py
    │   ├── cara_engine.py
    │   ├── game_theory.py
    │   ├── maneuver.py
    │   ├── threat_analysis.py
    │   ├── model_evaluation.py
    │   ├── __init__.py
    │   ├── xgboost_risk_model.pkl   ← eğitilmiş model
    │   ├── lstm_orbit_model.pt      ← eğitilmiş model
    │   └── lstm_scaler.pkl          ← normalizasyon
    ├── cache/                        ← TLE cache (otomatik)
    ├── socrates.csv                  ← CelesTrak conjunction verisi
    ├── run_k2_live.py               ← ana çalıştırıcı
    ├── test_k2.py                   ← test paketi
    └── TUA_SOPRANOS_Colab_Training.ipynb
```

---

## Çıktı Örneği

```
  Uydu             Tehdit              Mesafe       Pc      CARA
  ──────────────── ─────────────────── ──────── ──────── ──────
  Gokturk-2        OAO 1               310290m  4.32e-211  🟢 GREEN
  Turksat 1B       INMARSAT 2-F4       305368m  0.00e+00   🟢 GREEN
  Turksat 4B       COSMOS 775          796766m  0.00e+00   🟢 GREEN
```

---

## Hackathon — TUA SOPRANOS Ekibi
