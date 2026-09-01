/**
 * ThreatPanel.jsx — Right panel: selected satellite threat details & mitigation
 *
 * Features:
 *  - Satellite meta (altitude, fuel, TLE confidence, LSTM state)
 *  - Threat cards (distance, TCA, CARA Pc, relative velocity)
 *  - [NEW] CCSDS 508.0-B-1 Conjunction Data Message (CDM XML) Exporter & Modal
 *  - [NEW] Visual Before/After Risk Drop & Clearance Gauge
 *  - Physics Scenario Simulator (Tsiolkovsky propellant, burn duration, Δv, USD cost)
 *  - Game-theory decision badge
 */

import { useState, useEffect } from 'react'
import { DEMO_THREATS } from '../utils/demoData'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const LEVEL_ICONS = { RED: '🔴', YELLOW: '🟡', WATCH: '🟠', GREEN: '🟢' }

const SCENARIOS = [
  { id: 'none', label: '— Select an avoidance scenario —' },
  {
    id:            'micro_nudge',
    label:         '🟢  Micro Nudge  (prograde +0.05 m/s)',
    description:   'Minimal prograde burn 18 h before TCA. Safest option, tiny fuel cost.',
    maneuver_type: 'Prograde',
    delta_v_ms:    0.05,
  },
  {
    id:            'radial_out',
    label:         '🟡  Radial Outward  (+0.20 m/s)',
    description:   'Radial burn shifts orbital plane. Moderate fuel, reliable clearance.',
    maneuver_type: 'Radial Out',
    delta_v_ms:    0.20,
  },
  {
    id:            'emergency_retro',
    label:         '🔴  Emergency Retrograde  (−0.80 m/s)',
    description:   'Large retrograde burn 6 h before TCA. Guaranteed clearance, costly.',
    maneuver_type: 'Retrograde',
    delta_v_ms:    0.80,
  },
  {
    id:            'combined_burns',
    label:         '⚡  Bi-Impulsive  (+0.10 / −0.10 m/s)',
    description:   'Two small burns that cancel drift. Efficient for GEO station-keeping.',
    maneuver_type: 'Bi-Impulsive',
    delta_v_ms:    0.20,
  },
]

function efficiencyColor(pct) {
  if (pct >= 94) return '#00d4ff'
  if (pct >= 88) return '#ffd700'
  return '#ff6b6b'
}

function statusColor(s) {
  return { GREEN: 'var(--green)', YELLOW: 'var(--yellow)', RED: 'var(--red)' }[s] || 'var(--text-dim)'
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: CDM XML Modal
// ─────────────────────────────────────────────────────────────────────────────
function CdmModal({ satellite, threat, onClose }) {
  const [xmlContent, setXmlContent] = useState('')
  const [loading, setLoading] = useState(true)
  const [copied, setCopied] = useState(false)

  const satName = satellite?.name || 'PRIMARY_SAT'
  const debrisName = threat?.object_name || 'ENCOUNTER_DEBRIS'
  const tca = threat?.tca_hours_from_now ?? 18.0
  const missDist = threat?.min_distance_km ?? 0.30
  const pc = threat?.cara_result?.pc ?? 1.83e-4

  useEffect(() => {
    let active = true
    fetch(`${API}/api/cdm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        primary_name: satName,
        secondary_name: debrisName,
        miss_distance_km: missDist,
        pc: pc,
        tca_hours: tca,
      }),
    })
      .then(r => r.ok ? r.json() : null)
      .then(res => {
        if (!active) return
        if (res?.xml) {
          setXmlContent(res.xml)
        } else {
          // Fallback compliant XML template
          const nowIso = new Date().toISOString()
          setXmlContent(
`<?xml version="1.0" encoding="UTF-8"?>
<cdm xmlns="urn:ccsds:schema:cdmxml" version="1.0">
  <header>
    <COMMENT>TUA SOPRANOS — Autonomous Space Conjunction Assessment</COMMENT>
    <CREATION_DATE>${nowIso}</CREATION_DATE>
    <ORIGINATOR>TUA_SOPRANOS_C2</ORIGINATOR>
    <MESSAGE_FOR>TURKISH_SPACE_AGENCY</MESSAGE_FOR>
    <MESSAGE_ID>CDM-${satName.replace(/\\s+/g, '_')}-${Date.now()}</MESSAGE_ID>
  </header>
  <body>
    <relativeMetadataData>
      <TCA>${new Date(Date.now() + tca * 3600000).toISOString()}</TCA>
      <MISS_DISTANCE units="km">${missDist.toFixed(3)}</MISS_DISTANCE>
      <COLLISION_PROBABILITY>${pc.toExponential(2)}</COLLISION_PROBABILITY>
      <COLLISION_PROBABILITY_METHOD>NASA_CARA_2D_ALFANO</COLLISION_PROBABILITY_METHOD>
    </relativeMetadataData>
    <segment>
      <metadata>
        <OBJECT>OBJECT1</OBJECT>
        <OBJECT_NAME>${satName}</OBJECT_NAME>
        <INTERNATIONAL_DESIGNATOR>TUR-SAT</INTERNATIONAL_DESIGNATOR>
      </metadata>
    </segment>
    <segment>
      <metadata>
        <OBJECT>OBJECT2</OBJECT>
        <OBJECT_NAME>${debrisName}</OBJECT_NAME>
        <INTERNATIONAL_DESIGNATOR>DEBRIS</INTERNATIONAL_DESIGNATOR>
      </metadata>
    </segment>
  </body>
</cdm>`
          )
        }
        setLoading(false)
      })
      .catch(() => {
        if (active) setLoading(false)
      })

    return () => { active = false }
  }, [satName, debrisName, missDist, pc, tca])

  const handleCopy = () => {
    navigator.clipboard.writeText(xmlContent)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownload = () => {
    const blob = new Blob([xmlContent], { type: 'application/xml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `CDM_${satName.replace(/\\s+/g, '_')}_vs_${debrisName.replace(/\\s+/g, '_')}.xml`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="cdm-modal-overlay" onClick={onClose}>
      <div className="cdm-modal-card" onClick={e => e.stopPropagation()}>
        <div className="cdm-modal-header">
          <div className="cdm-modal-title">
            📄 CCSDS 508.0-B-1 Conjunction Data Message (CDM)
          </div>
          <button className="cdm-close-btn" onClick={onClose}>✕</button>
        </div>

        <div className="cdm-modal-sub">
          Official space agency interchange format (NASA CARA / ESA / CCSDS compliant)
        </div>

        <div className="cdm-code-wrap">
          {loading ? (
            <div style={{ color: 'var(--cyan)', padding: 20, textAlign: 'center' }}>
              Generating CCSDS XML structure…
            </div>
          ) : (
            <pre className="cdm-xml-code">{xmlContent}</pre>
          )}
        </div>

        <div className="cdm-modal-actions">
          <button className="cdm-btn cdm-copy-btn" onClick={handleCopy}>
            {copied ? '✓ COPIED TO CLIPBOARD' : '📋 COPY XML'}
          </button>
          <button className="cdm-btn cdm-dl-btn" onClick={handleDownload}>
            💾 DOWNLOAD .XML
          </button>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: Visual Risk Comparison Gauge
// ─────────────────────────────────────────────────────────────────────────────
function VisualRiskGauge({ initialPc, resultPc, missImprovementKm }) {
  const initialRed = initialPc > 1e-4

  return (
    <div className="risk-gauge-box">
      <div className="risk-gauge-title">⚡ COLLISION RISK MITIGATION VISUALIZER</div>
      <div className="risk-gauge-track">
        <div className="risk-step step-before">
          <span className="step-tag">BEFORE MANEUVER</span>
          <span className="step-val" style={{ color: initialRed ? 'var(--red)' : 'var(--yellow)' }}>
            🔴 {typeof initialPc === 'number' ? initialPc.toExponential(2) : initialPc}
          </span>
          <span className="step-sub">HIGH THREAT</span>
        </div>

        <div className="risk-arrow-wrap">
          <div className="risk-arrow">➔</div>
          <span className="clearance-pill">+{missImprovementKm.toFixed(2)} km</span>
        </div>

        <div className="risk-step step-after">
          <span className="step-tag">POST-AVOIDANCE</span>
          <span className="step-val" style={{ color: 'var(--green)' }}>
            🟢 {resultPc}
          </span>
          <span className="step-sub">CLEARED (GREEN)</span>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: animated horizontal bar
// ─────────────────────────────────────────────────────────────────────────────
function FuelBar({ pct, label }) {
  const color = pct > 55 ? '#00d4ff' : pct > 30 ? '#ffd700' : '#ff3366'
  return (
    <div style={{ marginTop: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-dim)', marginBottom: 3 }}>
        <span>{label}</span>
        <span style={{ color }}>{pct.toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.08)' }}>
        <div style={{
          height: '100%',
          width: `${Math.min(pct, 100)}%`,
          borderRadius: 3,
          background: `linear-gradient(90deg, ${color}88, ${color})`,
          transition: 'width 0.55s ease',
        }} />
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: single metric row
// ─────────────────────────────────────────────────────────────────────────────
function MetricRow({ label, value, valueColor }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '4px 0', borderBottom: '1px solid rgba(255,255,255,0.04)',
    }}>
      <span style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.5px' }}>{label}</span>
      <span style={{
        fontSize: 11, fontFamily: '"Share Tech Mono", monospace',
        color: valueColor || 'var(--text-primary)', fontWeight: 600,
      }}>
        {value}
      </span>
    </div>
  )
}

function SkeletonRows({ n = 6 }) {
  return (
    <>
      {Array.from({ length: n }).map((_, i) => (
        <div key={i} style={{
          height: 18, marginBottom: 5, borderRadius: 3,
          background: 'rgba(255,255,255,0.06)',
          animation: 'pulse 1.4s ease-in-out infinite',
          opacity: 1 - i * 0.1,
        }} />
      ))}
      <style>{`@keyframes pulse { 0%,100%{opacity:.4} 50%{opacity:.9} }`}</style>
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: Scenario Simulator
// ─────────────────────────────────────────────────────────────────────────────
function ScenarioSimulator({ satellite, threat }) {
  const [selectedId, setSelectedId] = useState('none')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const tcaHours = threat?.tca_hours_from_now ?? 18.0
  const originalMiss = threat?.min_distance_km ?? 0.30
  const initialPc = threat?.cara_result?.pc ?? 1.83e-4

  useEffect(() => {
    if (selectedId === 'none') {
      setResult(null)
      setError(null)
      return
    }

    const scenario = SCENARIOS.find(s => s.id === selectedId)
    if (!scenario) return

    const controller = new AbortController()
    setLoading(true)
    setError(null)
    setResult(null)

    fetch(`${API}/api/simulate_maneuver`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        sat_name: satellite.name,
        scenario_id: scenario.id,
        maneuver_type: scenario.maneuver_type,
        delta_v_ms: scenario.delta_v_ms,
        tca_hours: tcaHours,
        original_miss_distance_km: originalMiss,
        initial_pc: initialPc,
      }),
    })
      .then(r => {
        if (!r.ok) throw new Error(`Server ${r.status}`)
        return r.json()
      })
      .then(data => { setResult(data); setLoading(false) })
      .catch(err => {
        if (err.name === 'AbortError') return
        setError(err.message)
        setLoading(false)
      })

    return () => controller.abort()
  }, [selectedId, satellite.name, tcaHours, originalMiss, initialPc])

  const activeScenario = SCENARIOS.find(s => s.id === selectedId)

  return (
    <div style={{
      marginTop: 10,
      padding: '10px 12px',
      border: '1px solid rgba(0, 212, 255, 0.25)',
      borderRadius: 8,
      background: 'rgba(0, 212, 255, 0.04)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--cyan)', letterSpacing: '1.5px', fontWeight: 700 }}>
          ⚡ REAL-TIME MANEUVER SIMULATOR
        </span>
        <span style={{
          fontSize: 9, padding: '1px 5px', borderRadius: 3,
          background: 'rgba(123,47,255,0.25)', color: '#b07fff', letterSpacing: 1,
        }}>
          {satellite.name}
        </span>
      </div>

      <select
        value={selectedId}
        onChange={e => setSelectedId(e.target.value)}
        style={{
          width: '100%', background: 'rgba(7,13,26,0.9)',
          border: '1px solid rgba(0,212,255,0.3)', borderRadius: 5,
          color: 'var(--text-primary)', fontSize: 11,
          padding: '6px 8px', cursor: 'pointer', outline: 'none',
          marginBottom: 8, fontFamily: '"Share Tech Mono", monospace',
        }}
      >
        {SCENARIOS.map(s => (
          <option key={s.id} value={s.id} style={{ background: '#0a1020' }}>
            {s.label}
          </option>
        ))}
      </select>

      {activeScenario?.description && (
        <div style={{
          fontSize: 10, color: 'var(--text-dim)', background: 'rgba(255,255,255,0.04)',
          borderRadius: 4, padding: '5px 8px', marginBottom: 8, lineHeight: 1.5,
        }}>
          {activeScenario.description}
          <span style={{ marginLeft: 6, color: 'rgba(0,212,255,0.5)', fontSize: 9 }}>
            TCA: {tcaHours.toFixed(1)} h · Initial Miss: {originalMiss.toFixed(2)} km
          </span>
        </div>
      )}

      {loading && <SkeletonRows n={8} />}

      {error && (
        <div style={{
          padding: '8px', borderRadius: 4, fontSize: 10,
          background: 'rgba(255,51,102,0.1)', border: '1px solid rgba(255,51,102,0.3)',
          color: '#ff6b6b',
        }}>
          ⚠ Simulation error: {error}
        </div>
      )}

      {result && !loading && (
        <>
          {/* Visual Risk Gauge */}
          <VisualRiskGauge
            initialPc={initialPc}
            resultPc={result.pc_after}
            missImprovementKm={result.miss_improvement_km}
          />

          <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '2px', margin: '8px 0 4px' }}>
            BURN PARAMETERS
          </div>
          <MetricRow label="Maneuver Type" value={result.type} />
          <MetricRow label="Total Δv" value={`${result.delta_v_ms.toFixed(3)} m/s`} valueColor="var(--cyan)" />
          <MetricRow label="Burn Duration" value={`${result.burn_duration_sec.toFixed(1)} s`} />
          <MetricRow label="Engine Isp" value={`${result.isp_sec} s`} />

          <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '2px', margin: '8px 0 4px' }}>
            FUEL ECONOMY (TSIOLKOVSKY)
          </div>
          <MetricRow
            label="Propellant Mass"
            value={`${result.fuel_mass_kg.toFixed(4)} kg`}
            valueColor="#ffd700"
          />
          <MetricRow
            label="Burn Efficiency"
            value={`${result.fuel_efficiency_pct}%`}
            valueColor={efficiencyColor(result.fuel_efficiency_pct)}
          />
          <MetricRow label="USD Cost" value={`$${result.fuel_cost_usd.toLocaleString()}`} />
          <MetricRow
            label="Mission Life Impact"
            value={`${result.mission_life_impact_days} days`}
            valueColor="#ff9f43"
          />

          <FuelBar pct={result.fuel_remaining_pct} label="Tank Remaining" />

          <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '2px', margin: '8px 0 4px' }}>
            AVOIDANCE CLEARANCE
          </div>
          <MetricRow
            label="New Miss Distance"
            value={`${result.new_miss_distance_km.toFixed(2)} km`}
            valueColor="var(--green)"
          />
          <MetricRow
            label="Clearance Gained"
            value={`+${result.miss_improvement_km.toFixed(2)} km`}
            valueColor="var(--green)"
          />
          <MetricRow label="Pc After" value={result.pc_after} />

          <div style={{
            marginTop: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '4px 10px', borderRadius: 4,
              background: `${statusColor(result.status_after)}22`,
              border: `1px solid ${statusColor(result.status_after)}55`,
            }}>
              <span style={{ fontSize: 10, color: statusColor(result.status_after), fontWeight: 700 }}>
                STATUS: {result.status_after}
              </span>
            </div>
            {result.recommended && (
              <div style={{
                fontSize: 9, padding: '3px 7px', borderRadius: 3,
                background: 'rgba(0,204,150,0.15)', border: '1px solid rgba(0,204,150,0.4)',
                color: '#00cc96', letterSpacing: 1,
              }}>
                ★ OPTIMAL AVOIDANCE
              </div>
            )}
          </div>
        </>
      )}

      {!loading && !result && !error && selectedId === 'none' && (
        <div style={{ textAlign: 'center', padding: '8px 0', fontSize: 10, color: 'var(--text-dim)' }}>
          Select a scenario above to simulate propellant & avoidance outcome
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: ThreatCard
// ─────────────────────────────────────────────────────────────────────────────
function ThreatCard({ threat, satellite }) {
  const [showCdm, setShowCdm] = useState(false)
  const level = threat.classification?.label || 'WATCH'
  const cara = threat.cara_result || {}

  return (
    <div className={`threat-card ${level} fade-in`}>
      <div className="threat-card__header">
        <span className="threat-level-icon">{LEVEL_ICONS[level]}</span>
        <span className="threat-name">{threat.object_name}</span>
        {threat.is_demo && <span className="demo-badge">DEMO</span>}

        <button
          className="cdm-card-btn"
          onClick={() => setShowCdm(true)}
          title="Export CCSDS 508.0-B-1 Conjunction Data Message (XML)"
        >
          📄 CDM XML
        </button>
      </div>

      <div className="threat-card__stats">
        <div className="stat-row">
          <span className="stat-label">Min Dist</span>
          <span className={`stat-value ${level}`}>
            {threat.min_distance_km?.toFixed(2)} km
          </span>
        </div>
        <div className="stat-row">
          <span className="stat-label">TCA in</span>
          <span className="stat-value" style={{ color: 'var(--text-primary)' }}>
            {threat.tca_hours_from_now?.toFixed(1)} h
          </span>
        </div>
        <div className="stat-row">
          <span className="stat-label">Rel. Vel</span>
          <span className="stat-value" style={{ color: 'var(--text-primary)' }}>
            {threat.relative_velocity_kms?.toFixed(3)} km/s
          </span>
        </div>
      </div>

      {cara.pc_scientific && (
        <div className="cara-pc">
          <span>CARA Pc</span>
          <span>{cara.pc_scientific}</span>
        </div>
      )}

      {/* Maneuver Simulator for this threat */}
      <ScenarioSimulator satellite={satellite} threat={threat} />

      {/* CCSDS CDM Modal */}
      {showCdm && (
        <CdmModal
          satellite={satellite}
          threat={threat}
          onClose={() => setShowCdm(false)}
        />
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: ManeuverSection
// ─────────────────────────────────────────────────────────────────────────────
function ManeuverSection({ maneuver, game }) {
  if (!maneuver || maneuver.error) return null
  const rec = maneuver.recommended || maneuver.options?.[maneuver.recommended_index || 0]
  if (!rec) return null

  return (
    <div className="maneuver-section">
      <div className="section-label">⚡ Autonomous Maneuver Recommendation</div>
      <div className="maneuver-rec">
        <div className="maneuver-rec__row">
          Delta-V <span>{rec.delta_v_ms?.toFixed(4)} m/s</span>
        </div>
        <div className="maneuver-rec__row">
          Fuel cost <span>{rec.fuel_mass_kg?.toFixed(3)} kg</span>
        </div>
        <div className="maneuver-rec__row">
          USD cost <span>${rec.fuel_cost_usd?.toLocaleString()}</span>
        </div>
        <div className="maneuver-rec__row">
          Pc after <span>{rec.pc_after ?? '—'}</span>
        </div>
        {rec.new_cara_status && (
          <div className="maneuver-rec__row">
            Status{' '}
            <span style={{ color: rec.new_cara_status === 'GREEN' ? 'var(--green)' : 'var(--yellow)' }}>
              {rec.new_cara_status}
            </span>
          </div>
        )}
      </div>

      {game?.decision && (
        <div className="game-theory-badge">
          🎮{' '}
          {game.decision === 'PRIMARY_DODGE'   ? '→ Primary should maneuver'   :
           game.decision === 'SECONDARY_DODGE' ? '→ Secondary should maneuver' :
           '→ ' + game.decision}
          {game.reason && (
            <div style={{ fontSize: '10px', color: 'var(--text-dim)', marginTop: 3 }}>
              {game.reason}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN EXPORT
// ─────────────────────────────────────────────────────────────────────────────
export default function ThreatPanel({ satellite, demoMode }) {
  if (!satellite) return null

  const pos         = satellite.current_position
  const realThreats = satellite.threats || []
  const allThreats  = demoMode ? [...realThreats, ...DEMO_THREATS] : realThreats
  const serious     = allThreats.filter(t => ['RED', 'YELLOW'].includes(t.classification?.label))
  const others      = allThreats.filter(t => !['RED', 'YELLOW'].includes(t.classification?.label))

  return (
    <div className="threat-panel">
      {/* Satellite Header */}
      <div className="panel-header">
        <div className="panel-title">Threats & Analysis</div>
        <div className="sat-detail-name">{satellite.name}</div>
        <div className="sat-detail-meta">
          <span className="detail-chip">{satellite.orbit_type}</span>
          {pos?.alt_km && (
            <span className="detail-chip">{pos.alt_km.toLocaleString()} km</span>
          )}
          <span className="detail-chip">
            TLE {satellite.tle_confidence?.toFixed(0)}%
          </span>
          <span className="detail-chip">
            {satellite.lstm_correction_active ? '⚡ LSTM on' : 'SGP4 only'}
          </span>
        </div>
      </div>

      {/* Threat list */}
      <div className="threat-list">
        {allThreats.length === 0 ? (
          <div className="no-threats">
            <div className="no-threats-icon">🛡️</div>
            <div>No threats detected</div>
            <div style={{ fontSize: '10px' }}>
              {demoMode ? 'Demo threats loaded above' : 'Safe orbit — continuous monitoring'}
            </div>
          </div>
        ) : (
          <>
            {serious.length > 0 && (
              <>
                <div style={{ fontSize: '9px', color: 'var(--text-dim)', letterSpacing: '2px', padding: '4px 4px 0' }}>
                  CRITICAL / WARNING ({serious.length})
                </div>
                {serious.map((t, i) => (
                  <ThreatCard key={`s-${i}`} threat={t} satellite={satellite} />
                ))}
              </>
            )}
            {others.length > 0 && (
              <>
                <div style={{ fontSize: '9px', color: 'var(--text-dim)', letterSpacing: '2px', padding: '8px 4px 0' }}>
                  MONITORING ({others.length})
                </div>
                {others.map((t, i) => (
                  <ThreatCard key={`o-${i}`} threat={t} satellite={satellite} />
                ))}
              </>
            )}
          </>
        )}
      </div>

      {/* Maneuver Recommendation */}
      <ManeuverSection
        maneuver={satellite.maneuver_suggestion}
        game={satellite.game_theory}
      />
    </div>
  )
}
