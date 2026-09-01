"""
retrain_xgb.py — Mevcut ortamda XGBoost modelini yeniden egit ve kaydet.

18 ozellik extract_features() ile ayni sirada olmalidir:
  miss_distance_km, radial_km, intrack_km, crosstrack_km,
  relative_velocity_kms, mahalanobis_distance, combined_hbr_m,
  primary_altitude_km, secondary_rcs_m2, secondary_type_encoded,
  covariance_trace_km2, covariance_det_km4, time_to_tca_hours,
  tle_age_hours, miss_to_hbr_ratio, velocity_angle_deg,
  combined_sigma_km, energy_parameter
"""

import os
import sys
import pickle
from datetime import datetime
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── 18-ozellikli egitim verisi ──────────────────────────────────────────
# Her satir bir conjunction ornegi.
# Kolonlar: ml_model.py FEATURE_NAMES sirasina gore.
# Etiketler: 0=GREEN, 1=YELLOW, 2=RED (ml_model.CLASS_LABELS ile birebir uyumlu)

X = np.array([
    # RED (label=2) — cok yakin, yuksek hiz
    [0.05,  0.03,  0.02,  0.01, 14.5,  0.5,  5.0, 35786, 1.0, 0, 0.01, 0.0001,  2.0,  6.0,  10.0, 85.0, 0.10, 1050.0],
    [0.10,  0.06,  0.03,  0.02, 12.0,  0.7,  5.0,   640, 2.0, 1, 0.02, 0.0002,  4.0,  8.0,  20.0, 75.0, 0.14,  720.0],
    [0.08,  0.05,  0.02,  0.01, 13.5,  0.6,  5.0, 35786, 1.5, 0, 0.01, 0.0001,  1.5,  5.0,  16.0, 90.0, 0.12,  844.0],
    [0.15,  0.10,  0.05,  0.03, 11.0,  0.9,  6.0,  1200, 3.0, 1, 0.03, 0.0003,  3.0, 10.0,  25.0, 70.0, 0.17,  403.0],

    # YELLOW (label=1) — orta mesafe, orta hiz
    [1.50,  1.00,  0.60,  0.40,  5.0,  3.0,  5.0, 35786, 1.0, 0, 0.20, 0.0200, 12.0, 12.0, 300.0, 60.0, 0.45,   16.7],
    [2.00,  1.20,  0.80,  0.50,  4.0,  4.0,  5.0,   800, 2.0, 0, 0.30, 0.0300, 18.0, 14.0, 400.0, 55.0, 0.55,   10.0],
    [1.20,  0.80,  0.50,  0.30,  6.0,  2.5,  5.5, 35786, 1.5, 1, 0.15, 0.0150, 10.0, 11.0, 218.0, 65.0, 0.39,   25.0],
    [3.00,  2.00,  1.00,  0.80,  3.5,  5.0,  6.0,  1500, 2.5, 0, 0.40, 0.0400, 20.0, 16.0, 500.0, 50.0, 0.63,    6.1],

    # GREEN (label=0) — guvende, uzak mesafe
    [50.0,  30.0, 15.0,  10.0,  1.0, 50.0,  5.0, 35786, 1.0, 0, 5.00, 5.0000, 48.0, 24.0, 5000., 45.0, 3.54,    0.01],
    [80.0,  50.0, 25.0,  18.0,  0.5, 80.0,  5.0,   640, 2.0, 1, 8.00, 8.0000, 72.0, 36.0, 8000., 40.0, 4.47,    0.003],
    [35.0,  20.0, 12.0,   8.0,  1.5, 35.0,  5.0, 20000, 1.5, 0, 3.50, 3.5000, 36.0, 18.0, 3500., 50.0, 2.65,    0.032],
    [100.0, 60.0, 30.0,  20.0,  0.3, 100.0, 5.0, 35786, 3.0, 2, 10.0,10.0000, 96.0, 48.0,10000., 35.0, 5.00,    0.001],
])

y = np.array([
    2, 2, 2, 2,   # RED
    1, 1, 1, 1,   # YELLOW
    0, 0, 0, 0    # GREEN
])

# ── Egitim ───────────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss',
)
model.fit(X, y)

cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
accuracy  = float(model.score(X, y))

print(f"Train accuracy : {accuracy:.3f}")
print(f"CV mean/std    : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

model_data = {
    "model":         model,
    "accuracy":      accuracy,
    "cv_mean":       float(cv_scores.mean()),
    "cv_std":        float(cv_scores.std()),
    "trained_at":    datetime.utcnow().isoformat(),
    "n_samples":     len(y),
    "data_source":   "mock_retrain",
    "training_type": "mock_18feature",
}

_BASE = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(_BASE, "model", "xgboost_risk_model.pkl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(model_data, f)
print(f"Saved -> {out_path} (current env version)")
