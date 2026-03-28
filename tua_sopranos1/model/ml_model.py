"""
ml_model.py — Conjunction Risk Classification (XGBoost)
========================================================
Makine öğrenmesi ile çarpışma riski sınıflandırması.

Pipeline:
  K1 veri çeker → XGBoost hızlı tarama → cara_engine detaylı Pc
                   (milisaniye)            (sadece riskli olanlar)

Eğitim: Google Colab'da yapılıyor (TUA_SOPRANOS_ML_Training.py)
  - Space-Track'ten GERÇEK conjunction verisi çekilir
  - Her conjunction için GERÇEK Pc hesaplanır → label
  - XGBoost eğitilir → xgboost_risk_model.pkl kaydedilir
  - Bu dosya sadece INFERENCE (tahmin) yapar

Model: XGBoost Gradient Boosted Trees
  - Gerçek Space-Track verisiyle eğitildi
  - Tabular veri için SOTA (NeurIPS 2022, Grinsztajn et al.)
  - SHAP ile açıklanabilir

Referanslar:
  - Grinsztajn et al. (2022) NeurIPS
  - NASA CARA ML screening research (2022)

Yazar: K2 Algoritma Mühendisi
Proje: TUA SOPRANOS
"""

import numpy as np
import os
import pickle
from typing import Dict, List, Optional

try:
    from .cara_engine import compute_pc, assess_cara_status
except ImportError:
    from cara_engine import compute_pc, assess_cara_status

try:
    import xgboost as xgb
    HAS_ML = True
except ImportError:
    HAS_ML = False


# ============================================================
# SABİTLER
# ============================================================

FEATURE_NAMES = [
    "miss_distance_km", "radial_km", "intrack_km", "crosstrack_km",
    "relative_velocity_kms", "mahalanobis_distance", "combined_hbr_m",
    "primary_altitude_km", "secondary_rcs_m2", "secondary_type_encoded",
    "covariance_trace_km2", "covariance_det_km4", "time_to_tca_hours",
    "tle_age_hours", "miss_to_hbr_ratio", "velocity_angle_deg",
    "combined_sigma_km", "energy_parameter",
]

CLASS_LABELS = ["GREEN", "YELLOW", "RED"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_risk_model.pkl")


# ============================================================
# MODEL YÜKLEME
# ============================================================

_cached_model = None

def load_model(path: str = None) -> dict:
    """Colab'da eğitilmiş modeli yükle"""
    global _cached_model
    
    if _cached_model is not None:
        return _cached_model
    
    model_path = path or MODEL_PATH
    
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        _cached_model = pickle.load(f)
    
    return _cached_model


def get_model_info() -> dict:
    """Model hakkında bilgi"""
    data = load_model()
    if data is None:
        return {"status": "NOT_FOUND", "message": "Model dosyası bulunamadı. Colab'da eğitip model/ klasörüne koy."}
    
    return {
        "status": "READY",
        "accuracy": data.get("accuracy", 0),
        "cv_mean": data.get("cv_mean", 0),
        "cv_std": data.get("cv_std", 0),
        "trained_at": data.get("trained_at", "?"),
        "n_samples": data.get("n_samples", 0),
        "data_source": data.get("data_source", "unknown"),
        "training_type": data.get("training_type", "unknown"),
    }


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(conjunction_data: dict) -> np.ndarray:
    """
    K1'in conjunction verisinden ML feature'ları çıkarır.
    18 feature → modelin beklediği format.
    """
    
    miss_km = conjunction_data.get("min_distance_km", 10.0)
    radial = abs(conjunction_data.get("radial_km", miss_km * 0.5))
    intrack = abs(conjunction_data.get("intrack_km", miss_km * 0.3))
    crosstrack = abs(conjunction_data.get("crosstrack_km", miss_km * 0.2))
    rel_vel = conjunction_data.get("relative_speed_kms", 1.0)
    
    hbr_m = conjunction_data.get("combined_hbr_m", 5.0)
    hbr_km = conjunction_data.get("combined_hbr_km", hbr_m / 1000)
    
    cov_2d = conjunction_data.get("combined_covariance_2d", [[0.1, 0], [0, 0.1]])
    cov_arr = np.array(cov_2d)
    cov_trace = np.trace(cov_arr) if cov_arr.ndim == 2 else 0.1
    cov_det = np.linalg.det(cov_arr) if cov_arr.ndim == 2 else 0.01
    
    sigma = np.sqrt(max(cov_trace / 2, 1e-10))
    mahalanobis = miss_km / max(sigma, 1e-10)
    
    pos = conjunction_data.get("primary_pos_tca", [42164, 0, 0])
    altitude = np.sqrt(sum(p**2 for p in pos)) - 6371.0 if pos else 35786.0
    
    rcs_m2 = conjunction_data.get("secondary_rcs_m2", 1.0)
    sec_type = conjunction_data.get("secondary_type_encoded", 0)
    time_to_tca = conjunction_data.get("time_to_tca_hours", 24.0)
    tle_age = conjunction_data.get("tle_age_hours", 12.0)
    
    miss_to_hbr = miss_km / max(hbr_km, 1e-6)
    vel_angle = conjunction_data.get("velocity_angle_deg", 90.0)
    combined_sigma = sigma * np.sqrt(2)
    energy_param = (rel_vel**2) / (2 * max(miss_km, 1e-6))
    
    return np.array([
        miss_km, radial, intrack, crosstrack,
        rel_vel, mahalanobis, hbr_m, altitude,
        rcs_m2, sec_type, cov_trace, cov_det,
        time_to_tca, tle_age, miss_to_hbr,
        vel_angle, combined_sigma, energy_param
    ]).reshape(1, -1)


# ============================================================
# TAHMİN (INFERENCE)
# ============================================================

def predict_risk(conjunction_data: dict, model_data: dict = None) -> dict:
    """
    ML ile hızlı risk tahmini.
    
    Binlerce conjunction'ı saniyeler içinde tarar.
    Sadece YELLOW/RED olanlar detaylı Pc hesabına gönderilir.
    """
    
    if model_data is None:
        model_data = load_model()
    
    if model_data is None:
        return {
            "predicted_class": "UNKNOWN",
            "confidence_pct": 0,
            "error": "Model yüklenmedi — Colab'da eğitip xgboost_risk_model.pkl'i model/ klasörüne koy",
            "needs_detailed_pc": True,
        }
    
    model = model_data["model"]
    features = extract_features(conjunction_data)
    
    pred_idx = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]
    
    pred_class = CLASS_LABELS[int(pred_idx)]
    confidence = float(pred_proba[int(pred_idx)]) * 100
    
    return {
        "predicted_class": pred_class,
        "confidence_pct": round(confidence, 1),
        "probabilities": {
            "GREEN": round(float(pred_proba[0]) * 100, 1),
            "YELLOW": round(float(pred_proba[1]) * 100, 1),
            "RED": round(float(pred_proba[2]) * 100, 1),
        },
        "needs_detailed_pc": pred_class in ["YELLOW", "RED"],
        "recommendation": {
            "GREEN": "Güvenli — detaylı Pc hesabı gereksiz",
            "YELLOW": "Dikkat — cara_engine ile detaylı Pc hesaplayın",
            "RED": "TEHLİKE — acil Pc hesabı + manevra değerlendirmesi",
        }[pred_class],
    }


def batch_predict(conjunction_list: list, model_data: dict = None) -> list:
    """Birden fazla conjunction için toplu tahmin"""
    if model_data is None:
        model_data = load_model()
    
    results = []
    for conj in conjunction_list:
        result = predict_risk(conj, model_data)
        results.append(result)
    
    risk_order = {"RED": 0, "YELLOW": 1, "GREEN": 2, "UNKNOWN": 3}
    results.sort(key=lambda x: risk_order.get(x["predicted_class"], 3))
    return results


# ============================================================
# HYBRID PIPELINE (ML + FİZİK)
# ============================================================

def hybrid_risk_assessment(conjunction_data: dict, model_data: dict = None) -> dict:
    """
    HYBRID PIPELINE — ML ön tarama + fizik doğrulaması.
    
      1. XGBoost milisaniyede risk tahmini
      2. YELLOW/RED ise → cara_engine ile kesin Pc
      3. GREEN ise → atla, hızlı sonuç
    """
    
    ml_result = predict_risk(conjunction_data, model_data)
    
    result = {
        "ml_prediction": ml_result["predicted_class"],
        "ml_confidence": ml_result["confidence_pct"],
        "ml_probabilities": ml_result.get("probabilities", {}),
    }
    
    if ml_result["needs_detailed_pc"]:
        try:
            miss_2d = np.array(conjunction_data.get("miss_2d", [0.5, 0.3]))
            cov_2d = np.array(conjunction_data.get("combined_covariance_2d",
                                                     [[0.1, 0], [0, 0.1]]))
            hbr_km = conjunction_data.get("combined_hbr_km", 0.005)
            
            pc = compute_pc(miss_2d, cov_2d, hbr_km)
            cara = assess_cara_status(pc)
            
            result["physics_pc"] = cara["pc"]
            result["physics_pc_scientific"] = cara["pc_scientific"]
            result["physics_cara_status"] = cara["status"]
            result["ml_physics_agreement"] = (ml_result["predicted_class"] == cara["status"])
            result["final_status"] = cara["status"]
            result["final_method"] = "physics_verified"
        except Exception as e:
            result["physics_error"] = str(e)
            result["final_status"] = ml_result["predicted_class"]
            result["final_method"] = "ml_only"
    else:
        result["final_status"] = "GREEN"
        result["final_method"] = "ml_screened"
        result["physics_skipped"] = True
    
    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ML MODEL — Inference Test")
    print("=" * 60)
    
    info = get_model_info()
    print(f"\n  Model durumu: {info['status']}")
    
    if info["status"] == "READY":
        print(f"  Accuracy: {info['accuracy']*100:.1f}%")
        print(f"  Eğitim verisi: {info['data_source']}")
        print(f"  Örnek sayısı: {info['n_samples']}")
        
        # Test
        test = {
            "min_distance_km": 0.3,
            "radial_km": 0.1, "intrack_km": 0.2, "crosstrack_km": 0.1,
            "relative_speed_kms": 0.08,
            "combined_hbr_m": 8.0, "combined_hbr_km": 0.008,
            "combined_covariance_2d": [[0.04, 0.001], [0.001, 0.02]],
            "miss_2d": [0.15, 0.08],
            "time_to_tca_hours": 6.0,
        }
        
        pred = predict_risk(test)
        print(f"\n  Tahmin: {pred['predicted_class']} (%{pred['confidence_pct']})")
        
        hybrid = hybrid_risk_assessment(test)
        print(f"  Hybrid: {hybrid['final_status']} ({hybrid['final_method']})")
    else:
        print(f"  {info['message']}")
        print(f"\n  Çözüm:")
        print(f"  1. Google Colab'da TUA_SOPRANOS_ML_Training.py çalıştır")
        print(f"  2. xgboost_risk_model.pkl dosyasını indir")
        print(f"  3. model/ klasörüne koy")
    
    print(f"\n{'=' * 60}")
