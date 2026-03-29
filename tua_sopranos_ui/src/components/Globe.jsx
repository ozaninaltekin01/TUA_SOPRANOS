/**
 * Globe.jsx — CesiumJS 3D Dunya Gorunumu
 *
 * Gosterilen katmanlar (secili uyduda):
 *   • Tum uydular — kucuk renkli noktalar
 *   • Secili uydu — buyuk pulsing nokta + isim etiketi
 *   • SGP4 yolu   — mavi cizgi (200 nokta, duz)
 *   • LSTM yolu   — cyan kesik cizgi (daha parlak)
 *   • Tehditler   — renkli noktalar (RED/YEL/WATCH)
 *   • Tehdit yolları — renkli cizgiler
 *   • TCA cizgisi — beyaz kesikli (uydu → tehdit TCA)
 */

import { useEffect, useRef, useCallback } from 'react'
import * as Cesium from 'cesium'
import { ORBIT_LINE_COLORS, LEVEL_COLORS, DEMO_THREATS } from '../utils/demoData'

// Tum uydular icin uretilen kucuk nokta buyuklugu
const SAT_POINT_SIZE = 7
const SELECTED_SIZE  = 14

export default function Globe({ allSatellites, selectedSat, demoMode, onGlobeReady }) {
  const containerRef  = useRef(null)
  const viewerRef     = useRef(null)
  const pulseRef      = useRef(0)

  // ── Viewer kurulumu (bir kez) ──────────────────────────────
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return

    const viewer = new Cesium.Viewer(containerRef.current, {
      // Ion token yoksa Natural Earth II kullan (cesium paketi icinde geliyor)
      imageryProvider: new Cesium.TileMapServiceImageryProvider({
        url: Cesium.buildModuleUrl('Assets/Textures/NaturalEarthII'),
      }),
      baseLayerPicker:       false,
      geocoder:              false,
      homeButton:            false,
      sceneModePicker:       false,
      navigationHelpButton:  false,
      animation:             false,
      timeline:              false,
      fullscreenButton:      false,
      infoBox:               false,
      selectionIndicator:    false,
      creditContainer:       document.createElement('div'), // credits gizle
    })

    // Uzay arkaplan
    viewer.scene.backgroundColor = Cesium.Color.BLACK
    viewer.scene.globe.enableLighting = true
    viewer.scene.globe.atmosphereLightIntensity = 20.0

    // Atmosfer parlama efekti (gece tarafi)
    if (viewer.scene.skyAtmosphere) {
      viewer.scene.skyAtmosphere.show = true
    }

    // Baslangic kamera: Dunya tam gorunur
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(28.0, 10.0, 25_000_000),
    })

    viewerRef.current = viewer
    if (onGlobeReady) onGlobeReady()

    return () => {
      if (viewerRef.current && !viewerRef.current.isDestroyed()) {
        viewerRef.current.destroy()
        viewerRef.current = null
      }
    }
  }, []) // eslint-disable-line

  // ── Tum uydu noktalari — arka plan ────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || !allSatellites?.length) return

    // Eski arka plan noktalarini kaldir
    const remove = []
    viewer.entities.values.forEach(e => {
      if (e.id?.startsWith('bg-sat-')) remove.push(e.id)
    })
    remove.forEach(id => viewer.entities.removeById(id))

    allSatellites.forEach(sat => {
      const pos = sat.current_position
      if (!pos) return
      const level = sat.threat_level || 'GREEN'
      const color = LEVEL_COLORS[level] || LEVEL_COLORS.GREEN

      viewer.entities.add({
        id:       `bg-sat-${sat.name}`,
        position: Cesium.Cartesian3.fromDegrees(pos.lon ?? 0, pos.lat ?? 0, (pos.alt_km ?? 600) * 1000),
        point: {
          pixelSize:    SAT_POINT_SIZE,
          color:        color.withAlpha(0.7),
          outlineColor: Cesium.Color.WHITE.withAlpha(0.3),
          outlineWidth: 1,
          scaleByDistance: new Cesium.NearFarScalar(1e6, 1.5, 5e7, 0.4),
        },
      })
    })
  }, [allSatellites])

  // ── Secili uydu + tehditler ────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return

    // Onceki secim entitylerini temizle
    const remove = []
    viewer.entities.values.forEach(e => {
      if (e.id?.startsWith('sel-')) remove.push(e.id)
    })
    remove.forEach(id => viewer.entities.removeById(id))

    if (!selectedSat) return

    const pos = selectedSat.current_position
    if (!pos?.lon) return

    const satPos = Cesium.Cartesian3.fromDegrees(pos.lon, pos.lat, pos.alt_km * 1000)
    const level  = selectedSat.threat_level || 'GREEN'

    // ── Secili uydu noktasi (pulsing) ──────────────────────
    pulseRef.current = 0
    viewer.entities.add({
      id:       'sel-sat-point',
      position: satPos,
      point: {
        pixelSize: new Cesium.CallbackProperty(() => {
          pulseRef.current += 0.04
          return SELECTED_SIZE + 5 * Math.sin(pulseRef.current)
        }, false),
        color:        (LEVEL_COLORS[level] || LEVEL_COLORS.GREEN).withAlpha(0.95),
        outlineColor: Cesium.Color.WHITE.withAlpha(0.6),
        outlineWidth: 2,
      },
    })

    // ── Uydu etiketi ───────────────────────────────────────
    viewer.entities.add({
      id:       'sel-sat-label',
      position: satPos,
      label: {
        text:          selectedSat.name,
        font:          'bold 13px "Orbitron", sans-serif',
        fillColor:     Cesium.Color.fromCssColorString('#00d4ff'),
        outlineColor:  Cesium.Color.BLACK,
        outlineWidth:  3,
        style:         Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset:   new Cesium.Cartesian2(0, -24),
        distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 5e7),
        translucencyByDistance:   new Cesium.NearFarScalar(1e6, 1.0, 4e7, 0.4),
      },
    })

    // ── SGP4 yol cizcisi (mavi cizgi) ───────────────────────
    const orbitPath = selectedSat.orbit_path || []
    if (orbitPath.length > 1) {
      viewer.entities.add({
        id:       'sel-orbit-sgp4',
        polyline: {
          positions: orbitPath.map(p =>
            Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt_km * 1000)
          ),
          width:   2,
          material: new Cesium.PolylineGlowMaterialProperty({
            glowPower: 0.15,
            color:     ORBIT_LINE_COLORS.sgp4,
          }),
          arcType:  Cesium.ArcType.NONE,
          clampToGround: false,
        },
      })
    }

    // ── LSTM duzeltilmis yol (cyan kesik cizgi) ─────────────
    const lstmPath = selectedSat.predicted_path || []
    if (lstmPath.length > 1) {
      viewer.entities.add({
        id:       'sel-orbit-lstm',
        polyline: {
          positions: lstmPath.map(p =>
            Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt_km * 1000)
          ),
          width:   3,
          material: new Cesium.PolylineDashMaterialProperty({
            color:       ORBIT_LINE_COLORS.lstm,
            dashLength:  20,
            dashPattern: 0b1111110000000000,
          }),
          arcType: Cesium.ArcType.NONE,
        },
      })
    }

    // ── Tehditler ──────────────────────────────────────────
    const allThreats = demoMode
      ? [...(selectedSat.threats || []), ...DEMO_THREATS]
      : (selectedSat.threats || [])

    allThreats.forEach((threat, idx) => {
      const tLevel = threat.classification?.label || 'WATCH'
      const tColor = LEVEL_COLORS[tLevel] || LEVEL_COLORS.WATCH
      const tLine  = ORBIT_LINE_COLORS[tLevel] || ORBIT_LINE_COLORS.WATCH

      // Tehdit TCA pozisyonu
      const tp = threat.threat_position
      if (tp?.lon !== undefined && tp?.lat !== undefined) {
        const tPos = Cesium.Cartesian3.fromDegrees(tp.lon, tp.lat, tp.alt_km * 1000)

        // Nokta
        viewer.entities.add({
          id:       `sel-threat-point-${idx}`,
          position: tPos,
          point: {
            pixelSize:    tLevel === 'RED' ? 11 : 8,
            color:        tColor,
            outlineColor: Cesium.Color.WHITE.withAlpha(0.4),
            outlineWidth: 1.5,
          },
        })

        // Etiket
        viewer.entities.add({
          id:       `sel-threat-label-${idx}`,
          position: tPos,
          label: {
            text:         threat.object_name,
            font:         '11px "Share Tech Mono", monospace',
            fillColor:    tColor,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style:        Cesium.LabelStyle.FILL_AND_OUTLINE,
            pixelOffset:  new Cesium.Cartesian2(0, -20),
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 3e7),
          },
        })

        // TCA baglanti cizgisi (uydu → tehdit)
        const pp = threat.primary_position
        if (pp?.lon !== undefined) {
          const pPos = Cesium.Cartesian3.fromDegrees(pp.lon, pp.lat, pp.alt_km * 1000)
          viewer.entities.add({
            id:       `sel-tca-line-${idx}`,
            polyline: {
              positions: [pPos, tPos],
              width:     1.5,
              material:  new Cesium.PolylineDashMaterialProperty({
                color:      tColor.withAlpha(0.5),
                dashLength: 8,
              }),
              arcType: Cesium.ArcType.NONE,
            },
          })
        }
      }

      // Tehdit yuzugu / yolu
      const tOrbit = threat.orbit_path || []
      if (tOrbit.length > 1) {
        viewer.entities.add({
          id:       `sel-threat-orbit-${idx}`,
          polyline: {
            positions: tOrbit.map(p =>
              Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt_km * 1000)
            ),
            width:   tLevel === 'RED' ? 2 : 1.5,
            material: new Cesium.PolylineGlowMaterialProperty({
              glowPower: tLevel === 'RED' ? 0.25 : 0.1,
              color:     tLine.withAlpha(0.75),
            }),
            arcType: Cesium.ArcType.NONE,
          },
        })
      }
    })

    // ── Kamera — uyduya ucur ───────────────────────────────
    const isGeo = (selectedSat.orbit_type === 'GEO')
    const distAbove = isGeo ? 8_000_000 : 3_000_000

    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(
        pos.lon,
        pos.lat + (isGeo ? 5 : 15),
        pos.alt_km * 1000 + distAbove
      ),
      orientation: {
        heading: Cesium.Math.toRadians(0),
        pitch:   Cesium.Math.toRadians(-25),
        roll:    0,
      },
      duration: 2.5,
      easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
    })

  }, [selectedSat, demoMode]) // eslint-disable-line

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
}
