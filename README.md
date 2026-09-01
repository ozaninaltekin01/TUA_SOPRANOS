# 🛰️ TUA SOPRANOS
### Türk Uydu Çarpışma Risk Analizi ve Önleme Sistemi

> **TUA SOPRANOS** — Türkiye'nin aktif uydularını uzay enkazı ve diğer cisimlerden kaynaklanan çarpışma tehlikelerine karşı gerçek zamanlı izleyen, makine öğrenmesi destekli bir orbital güvenlik platformudur.

---

## 📌 Proje Özeti

Düşük Dünya yörüngesi (LEO) ve jeostasyoner yörüngede (GEO) binlerce aktif uydu ve enkaz nesnesi bulunmaktadır. Her yakın geçiş, milyonlarca dolarlık altyapıyı tehdit edebilir. TUA SOPRANOS bu riski otomatik olarak hesaplar, sınıflandırır ve operatörlere önlem önerir.

Sistem şu aşamalardan oluşur:

```
Space-Track API (TLE verisi)
        ↓
[K1] Veri Katmanı   → SGP4 yörünge yayılımı, tehdit taraması
        ↓
[K2] Model Katmanı  → XGBoost risk sınıfı, LSTM yörünge düzeltmesi,
                       NASA CARA çarpışma olasılığı, oyun teorisi kararı
        ↓
[K3] Arayüz         → FastAPI REST + React/CesiumJS 3B görselleştirme
```

---

## 🛸 İzlenen Uydular (13 Adet)

### GEO — Haberleşme
| Uydu | NORAD ID | Durum |
|------|----------|-------|
| Türksat 3A | 37160 | Aktif |
| Türksat 4A | 39522 | Aktif |
| Türksat 4B | 40319 | Aktif |
| Türksat 5A | 47306 | Aktif |
| Türksat 5B | 50212 | Aktif |
| Türksat 6A | 60233 | Aktif |

### LEO — Gözlem / Keşif
| Uydu | NORAD ID | Durum |
|------|----------|-------|
| Göktürk-1 | 41829 | Aktif |
| Göktürk-2 | 38331 | Aktif |
| İMECE | 55469 | Aktif |
| Türksat 3U | 39770 | Aktif |

### Emekli (Enkaz Statüsünde)
Türksat 1B · Türksat 1C · Türksat 2A

---

## ⚙️ Mimari ve Teknolojiler

### Backend — Python
| Teknoloji | Kullanım |
|-----------|---------|
| **FastAPI** | REST API sunucusu |
| **SGP4** | TLE tabanlı yörünge yayılımı |
| **XGBoost** | Risk sınıflandırma (GREEN / YELLOW / RED) |
| **PyTorch LSTM** | SGP4 artık düzeltme / yörünge tahmini |
| **SciPy** | NASA CARA 2B çarpışma olasılığı entegrasyonu |
| **NumPy** | Matris cebiri, koordinat dönüşümleri (ECI↔ECEF) |
| **Uvicorn** | ASGI uygulama sunucusu |

### Frontend — JavaScript
| Teknoloji | Kullanım |
|-----------|---------|
| **React 18** | Bileşen tabanlı arayüz |
| **Vite 5** | Geliştirme ve üretim derleme aracı |
| **CesiumJS** | 3B jeouzamsal küre görselleştirmesi |

### Veri Kaynakları
| Kaynak | İçerik |
|--------|--------|
| **Space-Track API** | Güncel NORAD TLE verileri |
| **CelesTrak SOCRATES** | 131.677 kayıtlık eğitim veri seti |
| **CCSDS 508.0 CDM** | Uluslararası standart çarpışma veri mesajı (XML) |

---

## 🔬 Temel Özellikler

### 1. Gerçek Zamanlı TLE Yönetimi
Space-Track API'den en güncel ephemerisleri çeker. Veri yaşına göre TLE güven skoru (0–100%) hesaplar ve akıllı önbellekleme ile gereksiz API çağrısını önler.

### 2. SGP4 Yörünge Yayılımı
Her uydu için anlık konum ve 24 saatlik yörünge yayı hesaplanır. ECI koordinat sisteminden ECEF ve coğrafi koordinatlara dönüşüm gerçekleştirilir.

### 3. NASA CARA 2B Çarpışma Olasılığı (Pc)
Foster (1992) yöntemine dayalı çift entegrasyon ile çarpışma olasılığı hesaplanır. Birleşik kovaryans matrisi, HBR (Hard Body Radius) ve göreli hız kullanılır.

| Pc Eşiği | Seviye | Eylem |
|----------|--------|-------|
| Pc ≥ 1×10⁻⁴ | 🔴 RED | Acil manevra |
| 1×10⁻⁶ ≤ Pc < 1×10⁻⁴ | 🟡 YELLOW | Manevra değerlendirmesi |
| Pc < 1×10⁻⁶ | 🟢 GREEN | İzleme |

### 4. Hibrit Makine Öğrenmesi Hattı
- **XGBoost**: 18 özellikli tablo verisiyle eğitilmiş çok sınıflı risk sınıflandırıcı. Binlerce yakın geçiş milisaniyeler içinde taranır; yalnızca YELLOW/RED olanlar CARA hesabına gönderilir.
- **LSTM**: İki katmanlı LSTM ağı, SGP4'ün uzun vadeli birikimli hatasını düzeltir ve daha hassas yörünge tahmini üretir.

### 5. Manevra Karar Sistemi
Tsiolkovsky füze denklemi kullanılarak üç manevra seçeneği hesaplanır:

```
m_yakıt = m_ıslak × (1 − exp(−|Δv| / (Isp × g₀)))
```

| Strateji | Δv | Avantaj |
|----------|----|---------|
| Micro Nudge (prograde) | ~0.05 m/s | Minimum yakıt tüketimi |
| Radyal İtme | ~0.20 m/s | Güvenilir açıklık |
| Acil Retrograde | ~0.80 m/s | Garantili kaçınma |
| Çift Vuruşlu | ~0.20 m/s | GEO istasyon tutma dostu |

### 6. Oyun Teorisi ile Karar Optimizasyonu
Nash denge analizi, manevrayı kimin yapması gerektiğine karar verir: birincil uydu mu (Türk uydusu) yoksa ikincil nesne mi? Yakıt bütçesi, çarpışma olasılığı ve manevra kapasitesi dikkate alınır.

### 7. CDM Üretimi
CCSDS 508.0 standardına uygun Conjunction Data Message (XML) otomatik üretilir. Operatörlere ve uluslararası ajanslarla paylaşıma hazır formatta çıktı verilir.

### 8. 3B Gerçek Zamanlı Gösterge Paneli
- **Küre**: ArcGIS World Imagery üzerine gerçekçi uydu fotoğrafları + Cesium Ion 3B arazi
- **Yörüngeler**: SGP4 yolu (beyaz), LSTM tahmini (cyan kesikli), tehdit yörüngesi (renk kodlu)
- **Tehdit noktaları**: TCA (Time of Closest Approach) konumu, mesafe çizgisi
- **Manevra Simülatörü**: Senaryo seçilince backend fizik hesabı çalışır, sonuçlar anlık gösterilir

---

## 🗂️ Proje Yapısı

```
TUA_SOPRANOS/
├── tua_sopranos1/                   # Python backend
│   ├── Veri_analizi/                # K1 — Veri ve yörünge katmanı
│   │   ├── config.py                # Uydu kayıt defteri, CARA eşikleri
│   │   ├── data_fetch.py            # Space-Track TLE çekimi ve önbellekleme
│   │   ├── orbit_calc.py            # SGP4, RIC koordinatları, kovaryans
│   │   └── threat_analysis.py       # Tehdit önceliklendirme, ıska mesafesi
│   ├── model/                       # K2 — Model katmanı
│   │   ├── cara_engine.py           # NASA CARA 2B Pc hesaplayıcı
│   │   ├── game_theory.py           # Nash denge manevra kararı
│   │   ├── maneuver.py              # Delta-V ve yakıt hesapları
│   │   ├── ml_model.py              # XGBoost + LSTM çıkarım motoru
│   │   ├── ml_training.py           # Model eğitim hattı
│   │   ├── xgboost_risk_model.pkl   # Eğitilmiş XGBoost modeli
│   │   ├── lstm_orbit_model.pt      # Eğitilmiş LSTM modeli
│   │   └── lstm_scaler.pkl          # LSTM normalizasyon parametreleri
│   ├── api.py                       # FastAPI REST sunucusu
│   ├── ui_data_generator.py         # Frontend JSON toplayıcı
│   ├── retrain_xgb.py               # XGBoost yeniden eğitim scripti
│   └── TUA_SOPRANOS_Colab_Training.ipynb
│
├── tua_sopranos_ui/                 # React + CesiumJS frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Globe.jsx            # 3B CesiumJS küre
│   │   │   ├── SatellitePanel.jsx   # Sol panel — uydu listesi
│   │   │   ├── ThreatPanel.jsx      # Sağ panel — tehdit detayı + simülatör
│   │   │   └── StatusBar.jsx        # Üst durum çubuğu
│   │   ├── hooks/useApiData.js      # API polling hook
│   │   └── utils/demoData.js        # Demo veri ve renk sabitleri
│   ├── .env                         # VITE_API_URL, VITE_CESIUM_TOKEN
│   └── vite.config.js
│
└── CDM_Turksat_3U.xml               # Örnek CCSDS 508.0 CDM dosyası
```

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.10+
- Node.js 18+
- Space-Track hesabı (ücretsiz — [space-track.org](https://www.space-track.org))
- Cesium Ion hesabı (ücretsiz — [ion.cesium.com](https://ion.cesium.com))

### Backend

```bash
cd tua_sopranos1

# Bağımlılıkları yükle
pip install fastapi uvicorn sgp4 numpy scipy torch xgboost scikit-learn joblib

# XGBoost modelini mevcut ortam için yeniden eğit
python retrain_xgb.py

# Sunucuyu başlat
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API dokümantasyonu: [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend

```bash
cd tua_sopranos_ui

# Bağımlılıkları yükle
npm install

# .env dosyasını yapılandır
# VITE_API_URL=http://localhost:8000
# VITE_CESIUM_TOKEN=<Cesium Ion tokenınız>

# Geliştirme sunucusunu başlat
npm run dev
```

Arayüz: [http://localhost:3000](http://localhost:3000)

---

## 📡 API Uç Noktaları

| Yöntem | Uç Nokta | Açıklama |
|--------|----------|---------|
| GET | `/api/full` | Tüm uydu verisi (yörüngeler, tehditler, manevra önerileri) |
| GET | `/api/status` | Sistem özeti (hafif) |
| GET | `/api/satellites` | Uydu listesi |
| GET | `/api/satellite/{name}` | Tek uydu tam detayı |
| GET | `/api/threats/{name}` | Uydu tehdit listesi (seviye filtresi destekli) |
| POST | `/api/simulate_maneuver` | Fizik tabanlı manevra simülasyonu |
| POST | `/api/refresh` | Önbelleği zorla yenile |

---

## 📊 Veri Akışı

```
1.  TLE Çekimi          Space-Track API → önbellek
2.  SGP4 Yayılımı       Anlık konum + 24h yörünge yayı
3.  Enkaz Taraması      200 km eşiği içindeki tehdit adayları
4.  CARA Hesabı         Her tehdit için 2B Pc çift entegrasyonu
5.  XGBoost Sınıfı      GREEN / YELLOW / RED risk etiketi
6.  LSTM Düzeltmesi     SGP4 artık tahmini → rafine yörünge
7.  Manevra Seçenekleri 3 × Δv stratejisi + yakıt maliyeti
8.  Nash Dengesi        Kimin manevra yapacağı kararı
9.  CDM Dışa Aktarım    CCSDS 508.0 XML operatörlere
10. UI Güncelleme       REST API → React / CesiumJS
```

---

## 🏆 Hackathon

Bu proje **TUA (Türkiye Uzay Ajansı)** hackathonu kapsamında geliştirilmiştir.

**Takım:** TUA SOPRANOS

---

## 📄 Standartlar ve Referanslar

- **CCSDS 508.0-B-1** — Conjunction Data Message standardı
- **Foster (1992)** — 2D Pc hesaplama yöntemi (NASA CARA)
- **Grinsztajn et al. (2022)** — Tablo verisi için XGBoost üstünlüğü (NeurIPS)
- **Tsiolkovsky Roketi Denklemi** — Delta-V ve yakıt kütlesi ilişkisi
- **SGP4/SDP4** — NORAD standart yörünge yayılım modeli

---

<div align="center">
  <sub>Türk uydularını korumak için geliştirildi 🇹🇷</sub>
</div>
