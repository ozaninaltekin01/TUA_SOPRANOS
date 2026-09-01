/**
 * Globe.jsx — CesiumJS 3D Dunya Gorunumu (v2)
 *
 * Duzeltmeler:
 *   - GEO kamera: 70,000 km irtifadan tam GEO halkasi + uydu gorunuyor
 *   - LEO kamera: uyduyu ve yoru saran uygun zoom
 *   - GEO orbit cizgisi: 360-nokta matematiksel halka (SGP4 path GEO'da yuzey
 *     uzerinde sabit nokta gosterir, halka gostermez)
 *   - LEO orbit cizgisi: API'den gelen gercek 200-nokta SGP4 yolu
 *   - Kamera kontrolleri: daha akici inertia, zoom limitleri
 *   - viewer.flyTo(entity) kullanilarak otomatik bounding ile merkezleme
 */

import { useEffect, useRef } from 'react'
import * as Cesium from 'cesium'
import { ORBIT_LINE_COLORS, LEVEL_COLORS, DEMO_THREATS } from '../utils/demoData'

// Cesium Ion ucretsiz katman — Dunya Arazisi icin gerekli.
// https://ion.cesium.com adresinden ucretsiz hesap aciniz ve token'inizi asagiya yapistirin.
// Token olmadan arazi duz kalir (Ellipsoid), goruntu yine gercekci gorunur.
const ION_TOKEN = import.meta.env.VITE_CESIUM_TOKEN || ''
if (ION_TOKEN) Cesium.Ion.defaultAccessToken = ION_TOKEN

const SAT_POINT_SIZE = 7
const SELECTED_SIZE  = 14

// ── GEO icin tam yuzuk olustur ─────────────────────────────────────────────
// GEO uydu Dunya-sabit koordinatlarda yilboyunca neredeyse hareket etmez.
// Gorsellestirme icin matematiksel tam halka gerekli.
function buildGeoRingPositions(altKm, inclinationDeg = 0.05, nPoints = 360) {
  const positions = []
  for (let i = 0; i <= nPoints; i++) {
    const t   = (i / nPoints) * 2 * Math.PI
    const lat = inclinationDeg * Math.sin(t)
    // -180..+180 araliginda esit aralikli boylamlar
    const lon = (i / nPoints) * 360 - 180
    positions.push(Cesium.Cartesian3.fromDegrees(lon, lat, altKm * 1000))
  }
  return positions
}

// ── Cizgi renklerinden Cesium material uret ────────────────────────────────
function glowMaterial(color, glowPower = 0.15) {
  return new Cesium.PolylineGlowMaterialProperty({ glowPower, color })
}

function dashMaterial(color, dashLength = 20) {
  return new Cesium.PolylineDashMaterialProperty({
    color,
    dashLength,
    dashPattern: 0b1111110000000000,
  })
}

// ── Entity ID yardimcilari ─────────────────────────────────────────────────
function clearByPrefix(viewer, prefix) {
  const ids = viewer.entities.values
    .filter(e => e.id?.startsWith(prefix))
    .map(e => e.id)
  ids.forEach(id => viewer.entities.removeById(id))
}

export default function Globe({ allSatellites, selectedSat, demoMode, onGlobeReady }) {
  const containerRef = useRef(null)
  const viewerRef    = useRef(null)
  const pulseRef     = useRef(0)

  // ── Viewer kurulumu (bir kez) ────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || viewerRef.current) return

    // ── Cesium 1.104+ async API kullan ───────────────────────────────────
    // fromUrl() / fromWorldTerrain() — eski sync constructor'lar deprecated.
    const viewer = new Cesium.Viewer(containerRef.current, {
      // ArcGIS World Imagery — gercek uydu fotograflari, token gerekmez
      baseLayer: Cesium.ImageryLayer.fromProviderAsync(
        Cesium.ArcGisMapServerImageryProvider.fromUrl(
          'https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
        )
      ),
      // Arazi: Ion token varsa 3D yukseklik, yoksa duz elipsoid
      terrain: ION_TOKEN
        ? Cesium.Terrain.fromWorldTerrain()
        : undefined,
      baseLayerPicker:      false,
      geocoder:             false,
      homeButton:           false,
      sceneModePicker:      false,
      navigationHelpButton: false,
      animation:            false,
      timeline:             false,
      fullscreenButton:     false,
      infoBox:              false,
      selectionIndicator:   false,
      creditContainer:      document.createElement('div'),
    })

    // ── Gercekci atmosfer & aydinlatma ───────────────────────────────────
    viewer.scene.backgroundColor = Cesium.Color.BLACK

    // Gunes pozisyonuna gore dinamik yuzey aydinlatmasi
    viewer.scene.globe.enableLighting          = true
    viewer.scene.globe.atmosphereLightIntensity = 10.0
    viewer.scene.globe.showGroundAtmosphere     = true

    // Daha gercekci atmosfer renkleri
    const atm = viewer.scene.skyAtmosphere
    if (atm) {
      atm.show                     = true
      atm.atmosphereLightIntensity = 20.0
      atm.hueShift                 = 0.0
      atm.saturationShift          = 0.1
      atm.brightnessShift          = 0.0
    }

    // Uzay siyahi arka plan + yildizlar
    viewer.scene.skyBox.show = true
    viewer.scene.sun.show    = true
    viewer.scene.moon.show   = true

    // Yer alti derinlik testi — uydular glob tarafindan gizlenmez
    viewer.scene.globe.depthTestAgainstTerrain = false

    // Fog: uzak mesafede atmosfer gorunumu
    viewer.scene.fog.enabled         = true
    viewer.scene.fog.density         = 0.0001
    viewer.scene.fog.minimumBrightness = 0.15

    // ── Kamera kontrolleri — akici hissettirmek icin ─────────────────────
    const ctrl = viewer.scene.screenSpaceCameraController
    ctrl.enableRotate    = true
    ctrl.enableZoom      = true
    ctrl.enableTilt      = true
    ctrl.enableTranslate = false   // pan'i kapat, globe merkezli hissettir
    ctrl.inertiaRotate   = 0.90    // kayma inertia (0=yok, 1=sonsuz)
    ctrl.inertiaZoom     = 0.80
    ctrl.minimumZoomDistance = 200_000      // 200 km
    ctrl.maximumZoomDistance = 150_000_000  // 150,000 km — GEO tam halka gorunumu icin

    // Baslangic: Turkiye uzerinde orta zoom
    viewer.camera.setView({
      destination: Cesium.Cartesian3.fromDegrees(35.0, 15.0, 28_000_000),
      orientation: { heading: 0, pitch: Cesium.Math.toRadians(-50), roll: 0 },
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

  // ── Arka plan: tum uydu noktalari ───────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || !allSatellites?.length) return

    clearByPrefix(viewer, 'bg-sat-')

    allSatellites.forEach(sat => {
      const pos = sat.current_position
      if (!pos?.lon) return
      const color = LEVEL_COLORS[sat.threat_level || 'GREEN'] || LEVEL_COLORS.GREEN

      viewer.entities.add({
        id:       `bg-sat-${sat.name}`,
        position: Cesium.Cartesian3.fromDegrees(pos.lon, pos.lat, pos.alt_km * 1000),
        point: {
          pixelSize:       SAT_POINT_SIZE,
          color:           color.withAlpha(0.65),
          outlineColor:    Cesium.Color.WHITE.withAlpha(0.25),
          outlineWidth:    1,
          scaleByDistance: new Cesium.NearFarScalar(1e6, 1.5, 8e7, 0.3),
        },
      })
    })
  }, [allSatellites])

  // ── Secim: uydu + orbit + tehditler ─────────────────────────────────────
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return

    clearByPrefix(viewer, 'sel-')

    if (!selectedSat) return

    const pos = selectedSat.current_position
    if (!pos?.lon) return

    const isGeo  = selectedSat.orbit_type === 'GEO'
    const altKm  = pos.alt_km ?? (isGeo ? 35786 : 640)
    const satPos = Cesium.Cartesian3.fromDegrees(pos.lon, pos.lat, altKm * 1000)
    const level  = selectedSat.threat_level || 'GREEN'

    // ── 1. Uydu noktasi (pulsing) ──────────────────────────────────────
    pulseRef.current = 0
    viewer.entities.add({
      id:       'sel-sat-point',
      position: satPos,
      point: {
        pixelSize: new Cesium.CallbackProperty(() => {
          pulseRef.current += 0.04
          return SELECTED_SIZE + 5 * Math.sin(pulseRef.current)
        }, false),
        color:        (LEVEL_COLORS[level] || LEVEL_COLORS.GREEN).withAlpha(0.98),
        outlineColor: Cesium.Color.WHITE.withAlpha(0.7),
        outlineWidth: 2,
        disableDepthTestDistance: Number.POSITIVE_INFINITY, // her zaman gorulur
      },
    })

    // ── 2. Uydu etiketi ────────────────────────────────────────────────
    viewer.entities.add({
      id:       'sel-sat-label',
      position: satPos,
      label: {
        text:         selectedSat.name,
        font:         'bold 13px "Orbitron", monospace',
        fillColor:    Cesium.Color.fromCssColorString('#00d4ff'),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style:        Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset:  new Cesium.Cartesian2(0, -26),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
        distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 8e7),
      },
    })

    // ── 3. Orbit cizgisi ───────────────────────────────────────────────
    //
    // GEO: Matematiksel tam halka (ekvatoral daire, ~35786 km irtifa)
    //      API orbit_path GEO icin cok kucuk bir kume verir (sabit uydu).
    //
    // LEO: Gercek SGP4 orbit_path (200 nokta, 24h yay — guzel gorunur)
    //
    let orbitPositions
    if (isGeo) {
      orbitPositions = buildGeoRingPositions(altKm, 0.05)
    } else {
      const orbitPath = selectedSat.orbit_path || []
      orbitPositions  = orbitPath.map(p =>
        Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt_km * 1000)
      )
    }

    if (orbitPositions.length > 1) {
      viewer.entities.add({
        id:       'sel-orbit-sgp4',
        polyline: {
          positions:  orbitPositions,
          width:      isGeo ? 2.5 : 2,
          material:   glowMaterial(ORBIT_LINE_COLORS.sgp4.withAlpha(isGeo ? 0.85 : 0.9), isGeo ? 0.25 : 0.15),
          arcType:    Cesium.ArcType.NONE,
        },
      })
    }

    // ── 4. LSTM tahmini yolu ───────────────────────────────────────────
    // GEO: Mock LSTM buyuk artiklar uretir → tam yuzuk cizer (~265,000 km).
    // Span dogrulamasi: GEO icin 5,000 km esigini asan yolu atla.
    const lstmPath = selectedSat.predicted_path || []
    if (lstmPath.length > 1) {
      let lstmSpanKm = 0
      for (let i = 1; i < lstmPath.length; i++) {
        const dlat = (lstmPath[i].lat - lstmPath[i - 1].lat) * 111
        const dlon = (lstmPath[i].lon - lstmPath[i - 1].lon) * 111 *
          Math.cos(lstmPath[i].lat * Math.PI / 180)
        lstmSpanKm += Math.sqrt(dlat * dlat + dlon * dlon)
      }
      const lstmValid = isGeo ? lstmSpanKm < 5_000 : true

      if (lstmValid) {
        viewer.entities.add({
          id:       'sel-orbit-lstm',
          polyline: {
            positions: lstmPath.map(p =>
              Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt_km * 1000)
            ),
            width:    3,
            material: dashMaterial(ORBIT_LINE_COLORS.lstm, 22),
            arcType:  Cesium.ArcType.NONE,
          },
        })
      }
    }

    // ── 5. Tehditler ───────────────────────────────────────────────────
    const allThreats = demoMode
      ? [...(selectedSat.threats || []), ...DEMO_THREATS]
      : (selectedSat.threats || [])

    allThreats.forEach((threat, idx) => {
      const tLevel = threat.classification?.label || 'WATCH'
      const tColor = LEVEL_COLORS[tLevel]         || LEVEL_COLORS.WATCH
      const tLine  = ORBIT_LINE_COLORS[tLevel]    || ORBIT_LINE_COLORS.WATCH

      // Tehdit TCA noktasi
      const tp = threat.threat_position
      if (tp?.lon !== undefined) {
        const tPos = Cesium.Cartesian3.fromDegrees(tp.lon, tp.lat, tp.alt_km * 1000)

        viewer.entities.add({
          id:       `sel-threat-pt-${idx}`,
          position: tPos,
          point: {
            pixelSize:   tLevel === 'RED' ? 13 : 9,
            color:       tColor,
            outlineColor: Cesium.Color.WHITE.withAlpha(0.5),
            outlineWidth: 2,
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
          },
        })

        viewer.entities.add({
          id:       `sel-threat-lbl-${idx}`,
          position: tPos,
          label: {
            text:         `${tLevel === 'RED' ? '⚠ ' : ''}${threat.object_name}`,
            font:         '11px "Share Tech Mono", monospace',
            fillColor:    tColor,
            outlineColor: Cesium.Color.BLACK,
            outlineWidth: 2,
            style:        Cesium.LabelStyle.FILL_AND_OUTLINE,
            pixelOffset:  new Cesium.Cartesian2(0, -22),
            disableDepthTestDistance: Number.POSITIVE_INFINITY,
            distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0, 5e7),
          },
        })

        // TCA baglanti cizgisi
        const pp = threat.primary_position
        if (pp?.lon !== undefined) {
          viewer.entities.add({
            id:       `sel-tca-${idx}`,
            polyline: {
              positions: [
                Cesium.Cartesian3.fromDegrees(pp.lon, pp.lat, pp.alt_km * 1000),
                tPos,
              ],
              width:    1.5,
              material: dashMaterial(tColor.withAlpha(0.5), 8),
              arcType:  Cesium.ArcType.NONE,
            },
          })
        }
      }

      // Tehdit orbit halkasi / yolu
      const tOrbit = threat.orbit_path || []
      if (tOrbit.length > 1) {
        viewer.entities.add({
          id:       `sel-threat-orbit-${idx}`,
          polyline: {
            positions: tOrbit.map(p =>
              Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt_km * 1000)
            ),
            width:    tLevel === 'RED' ? 2 : 1.5,
            material: glowMaterial(tLine.withAlpha(0.7), tLevel === 'RED' ? 0.3 : 0.1),
            arcType:  Cesium.ArcType.NONE,
          },
        })
      }
    })

    // ── 6. Kamera: orbit merkezli flyTo ───────────────────────────────
    //
    // GEO: 70,000 km irtifadan — GEO halkasi + uydu konumu net gorunur.
    //      Uydu sabit oldugu icin halkanin uzerinde parlak nokta gorulur.
    //
    // LEO: Uydu irtifasinin ~3000 km ustunden — orbit yayini net goster.
    //
    if (isGeo) {
      // lat=0 kamerasi halkayi KENAR KENAR (ince cizgi) gosterir.
      // Duzeltme: kamerayi 55 derece kuzeye al, Dunya merkezine dogru bak.
      // Bu sekilde GEO halkasi elips olarak gorunur (Saturn halkalari gibi).
      //
      // heading/pitch yerine direction+up kullan — ekvatorda heading/pitch
      // unreliable olur (gimbal lock benzeri). direction/up her zaman calisir.
      const camPos = Cesium.Cartesian3.fromDegrees(pos.lon, 55, 85_000_000)

      // Kamera → Dunya merkezi yonu
      const dir = Cesium.Cartesian3.normalize(
        Cesium.Cartesian3.subtract(Cesium.Cartesian3.ZERO, camPos, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      )
      // "Yukari" yonu: Kuzey Kutbu yonunu referans al
      const northPole = new Cesium.Cartesian3(0, 0, 1)
      const right = Cesium.Cartesian3.normalize(
        Cesium.Cartesian3.cross(dir, northPole, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      )
      const up = Cesium.Cartesian3.normalize(
        Cesium.Cartesian3.cross(right, dir, new Cesium.Cartesian3()),
        new Cesium.Cartesian3()
      )

      // flyTo animates movement; orientation arg is ignored during animation.
      // Apply the real direction+up in the complete callback via setView.
      viewer.camera.flyTo({
        destination:    camPos,
        duration:       2.5,
        easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT,
        complete: () => {
          viewer.camera.setView({ orientation: { direction: dir, up } })
        },
      })
    } else {
      // LEO: orbit yayini saran bounding sphere hesapla
      const orbitPath = selectedSat.orbit_path || []
      if (orbitPath.length > 1) {
        const bsPositions = orbitPath.map(p =>
          Cesium.Cartesian3.fromDegrees(p.lon, p.lat, p.alt_km * 1000)
        )
        const sphere = Cesium.BoundingSphere.fromPoints(bsPositions)
        viewer.camera.flyToBoundingSphere(sphere, {
          duration: 2.5,
          offset: new Cesium.HeadingPitchRange(
            Cesium.Math.toRadians(0),
            Cesium.Math.toRadians(-35),
            sphere.radius * 1.8,
          ),
        })
      } else {
        // orbit_path yoksa uyduya yaklas
        viewer.camera.flyTo({
          destination: Cesium.Cartesian3.fromDegrees(
            pos.lon, pos.lat + 15,
            (altKm + 3500) * 1000
          ),
          orientation: {
            heading: 0,
            pitch:   Cesium.Math.toRadians(-30),
            roll:    0,
          },
          duration: 2.5,
        })
      }
    }

  }, [selectedSat, demoMode]) // eslint-disable-line

  // Secim kaldirilinca Dunya gorunumune don
  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer || selectedSat) return

    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(35.0, 15.0, 28_000_000),
      orientation: { heading: 0, pitch: Cesium.Math.toRadians(-50), roll: 0 },
      duration: 1.8,
    })
  }, [selectedSat])

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
}
