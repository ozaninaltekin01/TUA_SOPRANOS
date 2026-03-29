/**
 * ThreatPanel.jsx — Sag panel, secili uydunun tehdit detaylari
 *
 * - Uydu bilgileri (irtifa, yakit, TLE guveni)
 * - Tehdit kartlari (mesafe, TCA, CARA Pc, XGBoost)
 * - Manevra onerisi (RED/YELLOW icin)
 * - Oyun teorisi karari
 */

import { DEMO_THREATS } from '../utils/demoData'

const LEVEL_ICONS  = { RED: '🔴', YELLOW: '🟡', WATCH: '🟠', GREEN: '🟢' }
const LEVEL_LABELS = { RED: 'CRITICAL', YELLOW: 'WARNING', WATCH: 'MONITOR', GREEN: 'SAFE' }

function ThreatCard({ threat }) {
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
        <div className="stat-row">
          <span className="stat-label">XGBoost</span>
          <span className={`stat-value ${threat.xgb_risk || 'GREEN'}`}>
            {threat.xgb_risk || '—'}
          </span>
        </div>
      </div>

      {cara.pc_scientific && (
        <div className="cara-pc">
          <span>CARA Pc</span>
          <span>{cara.pc_scientific}</span>
        </div>
      )}
    </div>
  )
}

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
            Status <span style={{ color: rec.new_cara_status === 'GREEN' ? 'var(--green)' : 'var(--yellow)' }}>
              {rec.new_cara_status}
            </span>
          </div>
        )}
      </div>

      {game?.decision && (
        <div className="game-theory-badge">
          🎮 {game.decision === 'PRIMARY_DODGE' ? '→ Primary should maneuver' :
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

export default function ThreatPanel({ satellite, demoMode }) {
  if (!satellite) return null

  const pos         = satellite.current_position
  const realThreats = satellite.threats || []
  const allThreats  = demoMode ? [...realThreats, ...DEMO_THREATS] : realThreats
  const serious     = allThreats.filter(t =>
    ['RED', 'YELLOW'].includes(t.classification?.label)
  )
  const others = allThreats.filter(t =>
    !['RED', 'YELLOW'].includes(t.classification?.label)
  )

  return (
    <div className="threat-panel">
      {/* Ust bilgi */}
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

      {/* Tehdit listesi */}
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
                {serious.map((t, i) => <ThreatCard key={`s-${i}`} threat={t} />)}
              </>
            )}
            {others.length > 0 && (
              <>
                <div style={{ fontSize: '9px', color: 'var(--text-dim)', letterSpacing: '2px', padding: '8px 4px 0' }}>
                  MONITORING
                </div>
                {others.map((t, i) => <ThreatCard key={`o-${i}`} threat={t} />)}
              </>
            )}
          </>
        )}
      </div>

      {/* Manevra onerisi */}
      <ManeuverSection
        maneuver={satellite.maneuver_suggestion}
        game={satellite.game_theory}
      />
    </div>
  )
}
