/**
 * SatellitePanel.jsx — Sol panel, uydu kart listesi
 *
 * Her kart:
 *   • Renk noktasi (tehdit seviyesi)
 *   • Uydu adi
 *   • Orbit tipi rozeti (GEO/LEO)
 *   • Irtifa + hiz
 *   • Yakit cubugu
 */

const LEVEL_ICONS = { RED: '🔴', YELLOW: '🟡', WATCH: '🟠', GREEN: '🟢' }

function FuelBar({ fuelKg, maxFuel = 500 }) {
  const pct = Math.min((fuelKg / maxFuel) * 100, 100)
  return (
    <div className="fuel-bar-wrap">
      <div className="fuel-bar">
        <div className="fuel-bar__fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="fuel-label">{fuelKg} kg</span>
    </div>
  )
}

function SatCard({ sat, isSelected, onClick }) {
  const level   = sat.threat_level || 'GREEN'
  const pos     = sat.current_position
  const retired = sat.status === 'retired'

  return (
    <div
      className={`sat-card ${isSelected ? 'selected' : ''} ${retired ? 'retired' : ''} fade-in`}
      onClick={onClick}
      title={retired ? `${sat.name} — Emekli` : sat.name}
    >
      <div className="sat-card__row">
        <span className="sat-card__name">{sat.name}</span>
        <div className={`threat-dot ${level}`} />
      </div>

      <div className="sat-card__meta">
        <span className={`orbit-badge ${sat.orbit_type}`}>{sat.orbit_type}</span>
        {pos && (
          <>
            <span>{pos.alt_km?.toLocaleString()} km</span>
            {pos.speed_kms && <span>{pos.speed_kms} km/s</span>}
          </>
        )}
        {sat.threats?.length > 0 && (
          <span style={{ color: level === 'GREEN' ? 'var(--text-dim)' : `var(--${level.toLowerCase()})` }}>
            {sat.threats.length} threat{sat.threats.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {!retired && sat.fuel_kg !== undefined && (
        <FuelBar fuelKg={sat.fuel_kg} />
      )}
    </div>
  )
}

export default function SatellitePanel({ satellites, selected, onSelect, loading }) {
  const active  = satellites?.filter(s => s.status !== 'retired') || []
  const retired = satellites?.filter(s => s.status === 'retired') || []

  return (
    <div className="satellite-panel">
      <div className="panel-header">
        <div className="panel-title">Satellites</div>
        <div className="panel-subtitle">
          {loading ? 'Loading…' : `${active.length} active · ${retired.length} retired`}
        </div>
      </div>

      <div className="sat-list">
        {loading && !satellites && (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="sat-card" style={{ opacity: 0.3, height: 64 }} />
          ))
        )}

        {/* Aktif uydular */}
        {active.map(sat => (
          <SatCard
            key={sat.name}
            sat={sat}
            isSelected={selected === sat.name}
            onClick={() => onSelect(sat.name === selected ? null : sat.name)}
          />
        ))}

        {/* Ayirici */}
        {retired.length > 0 && (
          <div style={{
            padding: '6px 12px 2px',
            fontSize: '9px',
            color: 'var(--text-dim)',
            letterSpacing: '2px',
            textTransform: 'uppercase',
          }}>
            Retired
          </div>
        )}

        {/* Emekli uydular */}
        {retired.map(sat => (
          <SatCard
            key={sat.name}
            sat={sat}
            isSelected={selected === sat.name}
            onClick={() => onSelect(sat.name === selected ? null : sat.name)}
          />
        ))}
      </div>
    </div>
  )
}
