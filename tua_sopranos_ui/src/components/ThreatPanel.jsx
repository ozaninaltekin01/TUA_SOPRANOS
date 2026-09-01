/**
 * ThreatPanel.jsx — Right panel: selected satellite threat details
 *
 * Sections:
 *  - Satellite meta (altitude, fuel, TLE confidence)
 *  - Threat cards  (distance, TCA, CARA Pc)
 *  - Maneuver recommendation (RED / YELLOW only)
 *  - Game-theory decision badge
 *  - [NEW] Türksat 6A Scenario Simulator
 *          Dropdown → POST /api/simulate_maneuver → real physics response
 *          (Tsiolkovsky fuel, burn duration, miss improvement, Pc, cost)
 */

import { useState, useEffect } from 'react'
import { DEMO_THREATS } from '../utils/demoData'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const LEVEL_ICONS  = { RED: '🔴', YELLOW: '🟡', WATCH: '🟠', GREEN: '🟢' }

// ─────────────────────────────────────────────────────────────────────────────
// SCENARIO DEFINITIONS  (only inputs — computed values come from the backend)
// ─────────────────────────────────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────
function efficiencyColor(pct) {
  if (pct >= 94) return '#00d4ff'
  if (pct >= 88) return '#ffd700'
  return '#ff6b6b'
}

function statusColor(s) {
  return { GREEN: 'var(--green)', YELLOW: 'var(--yellow)', RED: 'var(--red)' }[s] || 'var(--text-dim)'
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

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: loading skeleton rows
// ─────────────────────────────────────────────────────────────────────────────
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
// Only renders when satellite name contains "6A"
// ─────────────────────────────────────────────────────────────────────────────
function ScenarioSimulator({ satellite, threat }) {
  const [selectedId, setSelectedId] = useState('none')
  const [result,     setResult]     = useState(null)    // backend response
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)

  // TCA and initial Pc come from the live threat card when available
  const tcaHours     = threat?.tca_hours_from_now     ?? 18.0
  const originalMiss = threat?.min_distance_km        ?? 0.30
  const initialPc    = threat?.cara_result?.pc        ?? 1.83e-4

  // ── Fetch from backend whenever the dropdown changes ───────────────────
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
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      signal:  controller.signal,
      body: JSON.stringify({
        sat_name:                  satellite.name,
        scenario_id:               scenario.id,
        maneuver_type:             scenario.maneuver_type,
        delta_v_ms:                scenario.delta_v_ms,
        tca_hours:                 tcaHours,
        original_miss_distance_km: originalMiss,
        initial_pc:                initialPc,
      }),
    })
      .then(r => {
        if (!r.ok) throw new Error(`Server ${r.status}`)
        return r.json()
      })
      .then(data => { setResult(data); setLoading(false) })
      .catch(err  => {
        if (err.name === 'AbortError') return
        setError(err.message)
        setLoading(false)
      })

    return () => controller.abort()   // cleanup on scenario change
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

      {/* ── Title row ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--cyan)', letterSpacing: '1.5px', fontWeight: 700 }}>
          🛰 MANEUVER SIMULATOR
        </span>
        <span style={{
          fontSize: 9, padding: '1px 5px', borderRadius: 3,
          background: 'rgba(123,47,255,0.25)', color: '#b07fff', letterSpacing: 1,
        }}>
          {satellite.name}
        </span>
      </div>

      {/* ── Dropdown ── */}
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

      {/* ── Scenario description ── */}
      {activeScenario?.description && (
        <div style={{
          fontSize: 10, color: 'var(--text-dim)', background: 'rgba(255,255,255,0.04)',
          borderRadius: 4, padding: '5px 8px', marginBottom: 8, lineHeight: 1.5,
        }}>
          {activeScenario.description}
          <span style={{ marginLeft: 6, color: 'rgba(0,212,255,0.5)', fontSize: 9 }}>
            TCA: {tcaHours.toFixed(1)} h · Miss: {originalMiss.toFixed(2)} km
          </span>
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {loading && <SkeletonRows n={8} />}

      {/* ── Error state ── */}
      {error && (
        <div style={{
          padding: '8px', borderRadius: 4, fontSize: 10,
          background: 'rgba(255,51,102,0.1)', border: '1px solid rgba(255,51,102,0.3)',
          color: '#ff6b6b',
        }}>
          ⚠ Backend error: {error}
        </div>
      )}

      {/* ── Results from backend ── */}
      {result && !loading && (
        <>
          {/* BURN PARAMETERS */}
          <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '2px', marginBottom: 4 }}>
            BURN PARAMETERS
          </div>
          <MetricRow label="Maneuver Type"  value={result.type} />
          <MetricRow label="Total Δv"       value={`${result.delta_v_ms.toFixed(3)} m/s`}       valueColor="var(--cyan)" />
          <MetricRow label="Burn Duration"  value={`${result.burn_duration_sec.toFixed(1)} s`} />
          <MetricRow label="Engine Isp"     value={`${result.isp_sec} s`} />

          {/* FUEL ECONOMY */}
          <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '2px', margin: '8px 0 4px' }}>
            FUEL ECONOMY
          </div>
          <MetricRow
            label="Propellant Used"
            value={`${result.fuel_mass_kg.toFixed(4)} kg`}
            valueColor="#ffd700"
          />
          <MetricRow
            label="Burn Efficiency"
            value={`${result.fuel_efficiency_pct}%`}
            valueColor={efficiencyColor(result.fuel_efficiency_pct)}
          />
          <MetricRow label="USD Equiv."     value={`$${result.fuel_cost_usd.toLocaleString()}`} />
          <MetricRow
            label="Mission Life −"
            value={`${result.mission_life_impact_days} days`}
            valueColor="#ff9f43"
          />

          <FuelBar pct={result.fuel_remaining_pct} label="Tank remaining (of capacity)" />

          {/* AVOIDANCE OUTCOME */}
          <div style={{ fontSize: 9, color: 'var(--text-dim)', letterSpacing: '2px', margin: '8px 0 4px' }}>
            AVOIDANCE OUTCOME
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

          {/* POST-MANEUVER STATUS */}
          <div style={{
            marginTop: 8, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          }}>
            <div style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '4px 10px', borderRadius: 4,
              background: `${statusColor(result.status_after)}22`,
              border:     `1px solid ${statusColor(result.status_after)}55`,
            }}>
              <span style={{ fontSize: 10, color: statusColor(result.status_after), fontWeight: 700 }}>
                POST-MANEUVER: {result.status_after}
              </span>
            </div>
            {result.recommended && (
              <div style={{
                fontSize: 9, padding: '3px 7px', borderRadius: 3,
                background: 'rgba(0,204,150,0.15)', border: '1px solid rgba(0,204,150,0.4)',
                color: '#00cc96', letterSpacing: 1,
              }}>
                ★ OPTIMAL
              </div>
            )}
          </div>
        </>
      )}

      {/* ── Empty state ── */}
      {!loading && !result && !error && selectedId === 'none' && (
        <div style={{ textAlign: 'center', padding: '10px 0', fontSize: 10, color: 'var(--text-dim)' }}>
          Select a scenario to compute maneuver metrics
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: ThreatCard
// ─────────────────────────────────────────────────────────────────────────────
function ThreatCard({ threat, satellite }) {
  const level = threat.classification?.label || 'WATCH'
  const cara  = threat.cara_result || {}

  return (
    <div className={`threat-card ${level} fade-in`}>
      <div className="threat-card__header">
        <span className="threat-level-icon">{LEVEL_ICONS[level]}</span>
        <span className="threat-name">{threat.object_name}</span>
        {threat.is_demo && <span className="demo-badge">DEMO</span>}
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

      <ScenarioSimulator satellite={satellite} threat={threat} />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-COMPONENT: ManeuverSection (API-driven, unchanged)
// ─────────────────────────────────────────────────────────────────────────────
function ManeuverSection({ maneuver, game }) {
  if (!maneuver || maneuver.error) return null
  const rec = maneuver.recommended || maneuver.options?.[maneuver.recommended_index || 0]
  if (!rec) return null

  return (
    <div className="maneuver-section">
      <div className="section-label">⚡ Maneuver Recommendation</div>
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
      {/* ── Satellite header ── */}
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

      {/* ── Threat list ── */}
      <div className="threat-list">
        {allThreats.length === 0 ? (
          <div className="no-threats">
            <div className="no-threats-icon">🛡️</div>
            <div>No threats detected</div>
            <div style={{ fontSize: '10px' }}>
              {demoMode ? 'Demo threats loaded above' : 'Safe orbit — monitoring'}
            </div>
          </div>
        ) : (
          <>
            {serious.length > 0 && (
              <>
                <div style={{ fontSize: '9px', color: 'var(--text-dim)', letterSpacing: '2px', padding: '4px 4px 0' }}>
                  CRITICAL / WARNING
                </div>
                {serious.map((t, i) => (
                  <ThreatCard key={`s-${i}`} threat={t} satellite={satellite} />
                ))}
              </>
            )}
            {others.length > 0 && (
              <>
                <div style={{ fontSize: '9px', color: 'var(--text-dim)', letterSpacing: '2px', padding: '8px 4px 0' }}>
                  MONITORING
                </div>
                {others.map((t, i) => (
                  <ThreatCard key={`o-${i}`} threat={t} satellite={satellite} />
                ))}
              </>
            )}
          </>
        )}
      </div>

      {/* ── API-driven maneuver recommendation ── */}
      <ManeuverSection
        maneuver={satellite.maneuver_suggestion}
        game={satellite.game_theory}
      />
    </div>
  )
}
