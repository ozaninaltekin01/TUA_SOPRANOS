/**
 * demoData.js — Demo mod icin iki dramatik tehdit
 *
 * Demo mod acikken bu tehditler secili uydunun gercek verisi
 * uzerine eklenir. Tamamen sahte ama gercekci GEO parametreleri.
 *
 * - DEMO-DEBRIS-001 : RED  — 0.3 km, TCA 6.2h
 * - COSMOS-DEMO-2143: YELLOW — 7.8 km, TCA 18.5h
 */

import * as Cesium from 'cesium'

// Cikis noktasi: Turksat 5A nominali (42 E, 0 N, 35786 km)
const GEO_ALT = 35786

/**
 * GEO yuzukleri (tam daire gorunumu) — CesiumJS polyline icin
 * Gercekciligi artirmak icin hafif inklinasyon + uzunluk kayma eklendi.
 */
function buildGeoRing(lonCenterDeg, inclinationDeg, altKm, nPoints = 200) {
  const path = []
  for (let i = 0; i <= nPoints; i++) {
    const t = (i / nPoints) * 2 * Math.PI
    // Analemma benzeri kucuk dalgalanma
    const lat = inclinationDeg * Math.sin(t)
    const lonOffset = inclinationDeg * 0.5 * Math.sin(2 * t)
    const lon = lonCenterDeg + lonOffset
    path.push({ lat: round(lat, 4), lon: round(lon, 4), alt_km: altKm })
  }
  return path
}

function round(v, d) { return Math.round(v * 10 ** d) / 10 ** d }

// ── Demo tehdit 1 — RED ───────────────────────────────────────
export const DEMO_RED_THREAT = {
  object_name:           'DEMO-DEBRIS-001',
  min_distance_km:       0.3,
  tca_hours_from_now:    6.2,
  orbit_type:            'GEO',
  altitude_km:           GEO_ALT - 3.5,
  relative_velocity_kms: 0.048,
  is_demo:               true,
  classification: {
    label:       'RED',
    color:       '#ff3366',
    show_in_ui:  true,
    run_cara:    true,
    description: 'Acil — CARA Pc > 1e-4, manevra degerlendirin',
  },
  primary_position: { lat: 0.12,  lon: 42.30, alt_km: GEO_ALT },
  threat_position:  { lat: 0.09,  lon: 42.31, alt_km: GEO_ALT - 3.5 },
  cara_result: {
    status:        'RED',
    pc:            1.83e-4,
    pc_scientific: '1.83e-04',
  },
  xgb_risk:  'RED',
  // Yuzuk gorunumu: 0.15 deg inklinasyon, nominale yakin
  orbit_path: buildGeoRing(42.5, 0.15, GEO_ALT - 3.5, 150),
}

// ── Demo tehdit 2 — YELLOW ────────────────────────────────────
export const DEMO_YELLOW_THREAT = {
  object_name:           'COSMOS-DEMO-2143',
  min_distance_km:       7.8,
  tca_hours_from_now:    18.5,
  orbit_type:            'GEO',
  altitude_km:           GEO_ALT + 4.1,
  relative_velocity_kms: 0.122,
  is_demo:               true,
  classification: {
    label:       'YELLOW',
    color:       '#ffd700',
    show_in_ui:  true,
    run_cara:    true,
    description: 'Izle — CARA Pc > 1e-5, plan hazirlayin',
  },
  primary_position: { lat: -0.05, lon: 42.15, alt_km: GEO_ALT },
  threat_position:  { lat: -0.38, lon: 42.22, alt_km: GEO_ALT + 4.1 },
  cara_result: {
    status:        'YELLOW',
    pc:            3.21e-5,
    pc_scientific: '3.21e-05',
  },
  xgb_risk:  'YELLOW',
  // 0.4 deg inklinasyon — hafifce egik yuzuk
  orbit_path: buildGeoRing(41.8, 0.40, GEO_ALT + 4.1, 150),
}

export const DEMO_THREATS = [DEMO_RED_THREAT, DEMO_YELLOW_THREAT]

// ── Cesium entity renkleri ────────────────────────────────────
export const LEVEL_COLORS = {
  RED:    Cesium.Color.fromCssColorString('#ff3366'),
  YELLOW: Cesium.Color.fromCssColorString('#ffd700'),
  WATCH:  Cesium.Color.fromCssColorString('#aaaacc'),
  GREEN:  Cesium.Color.fromCssColorString('#00ff88'),
}

export const ORBIT_LINE_COLORS = {
  sgp4:   Cesium.Color.fromCssColorString('#4488ff'),
  lstm:   Cesium.Color.fromCssColorString('#00d4ff'),
  RED:    Cesium.Color.fromCssColorString('#ff3366'),
  YELLOW: Cesium.Color.fromCssColorString('#ffd700'),
  WATCH:  Cesium.Color.fromCssColorString('#888899'),
  GREEN:  Cesium.Color.fromCssColorString('#00ff88'),
}
