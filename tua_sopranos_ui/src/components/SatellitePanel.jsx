import { useState, useMemo } from 'react'

const LEVEL_ICONS = { RED: '🔴', YELLOW: '🟡', WATCH: '🟠', GREEN: '🟢' }

function FuelBar({ fuelKg, maxFuel = 500 }) {
  const pct = Math.min((fuelKg / maxFuel) * 100, 100)
  const color = pct > 50 ? 'var(--cyan)' : pct > 20 ? 'var(--yellow)' : 'var(--red)'
  return (
    <div className="fuel-bar-wrap">
      <div className="fuel-bar">
        <div className="fuel-bar__fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="fuel-label">{fuelKg} kg</span>
    </div>
  )
}

function SatCard({ sat, isSelected, onClick }) {
  const level = sat.threat_level || 'GREEN'
  const pos = sat.current_position
  const retired = sat.status === 'retired'
  const threatCount = sat.threats?.length || 0

  return (
    <div
      className={`sat-card ${isSelected ? 'selected' : ''} ${retired ? 'retired' : ''} ${level !== 'GREEN' ? 'has-risk' : ''} fade-in`}
      onClick={onClick}
      title={retired ? `${sat.name} — Emekli` : sat.name}
    >
      <div className="sat-card__row">
        <span className="sat-card__name">{sat.name}</span>
        <div className={`threat-dot ${level}`} title={`Risk: ${level}`} />
      </div>

      <div className="sat-card__meta">
        <span className={`orbit-badge ${sat.orbit_type}`}>{sat.orbit_type}</span>
        {pos && (
          <>
            <span>{pos.alt_km?.toLocaleString()} km</span>
            {pos.speed_kms && <span>{pos.speed_kms} km/s</span>}
          </>
        )}
        {threatCount > 0 && (
          <span style={{ color: level === 'GREEN' ? 'var(--text-dim)' : `var(--${level.toLowerCase()})`, fontWeight: 600 }}>
            {threatCount} threat{threatCount !== 1 ? 's' : ''}
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
  const [filter, setFilter] = useState('ALL')
  const [search, setSearch] = useState('')

  const allSats = satellites || []
  const activeCount = allSats.filter(s => s.status !== 'retired').length
  const geoCount = allSats.filter(s => s.orbit_type === 'GEO' && s.status !== 'retired').length
  const leoCount = allSats.filter(s => s.orbit_type === 'LEO' && s.status !== 'retired').length
  const threatCount = allSats.filter(s => s.threat_level && s.threat_level !== 'GREEN').length

  const filteredSats = useMemo(() => {
    let list = allSats

    // Tab Filter
    if (filter === 'ACTIVE') list = list.filter(s => s.status !== 'retired')
    else if (filter === 'GEO') list = list.filter(s => s.orbit_type === 'GEO')
    else if (filter === 'LEO') list = list.filter(s => s.orbit_type === 'LEO')
    else if (filter === 'THREATS') list = list.filter(s => (s.threat_level && s.threat_level !== 'GREEN') || (s.threats && s.threats.length > 0))

    // Search Query
    if (search.trim()) {
      const q = search.toLowerCase()
      list = list.filter(s => s.name.toLowerCase().includes(q) || (s.orbit_type && s.orbit_type.toLowerCase().includes(q)))
    }

    return list
  }, [allSats, filter, search])

  const activeSats = filteredSats.filter(s => s.status !== 'retired')
  const retiredSats = filteredSats.filter(s => s.status === 'retired')

  return (
    <div className="satellite-panel">
      {/* Header */}
      <div className="panel-header">
        <div className="panel-title">Fleet Command</div>
        <div className="panel-subtitle">
          {loading ? 'Scanning fleet…' : `${activeCount} Active · ${allSats.length - activeCount} Retired`}
        </div>
      </div>

      {/* Search Input */}
      <div className="sat-search-wrap">
        <span className="search-icon">🔍</span>
        <input
          type="text"
          className="sat-search-input"
          placeholder="Filter satellites..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
        {search && (
          <button className="search-clear-btn" onClick={() => setSearch('')}>✕</button>
        )}
      </div>

      {/* Filter Chips */}
      <div className="sat-filter-chips">
        <button
          className={`filter-chip ${filter === 'ALL' ? 'active' : ''}`}
          onClick={() => setFilter('ALL')}
        >
          ALL ({allSats.length})
        </button>
        <button
          className={`filter-chip ${filter === 'ACTIVE' ? 'active' : ''}`}
          onClick={() => setFilter('ACTIVE')}
        >
          ACTIVE ({activeCount})
        </button>
        <button
          className={`filter-chip ${filter === 'GEO' ? 'active' : ''}`}
          onClick={() => setFilter('GEO')}
        >
          GEO ({geoCount})
        </button>
        <button
          className={`filter-chip ${filter === 'LEO' ? 'active' : ''}`}
          onClick={() => setFilter('LEO')}
        >
          LEO ({leoCount})
        </button>
        <button
          className={`filter-chip threat-chip ${filter === 'THREATS' ? 'active' : ''}`}
          onClick={() => setFilter('THREATS')}
        >
          🚨 RISKS ({threatCount})
        </button>
      </div>

      {/* Satellite List */}
      <div className="sat-list">
        {loading && allSats.length === 0 && (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="sat-card" style={{ opacity: 0.3, height: 64 }} />
          ))
        )}

        {/* Active sats */}
        {activeSats.map(sat => (
          <SatCard
            key={sat.name}
            sat={sat}
            isSelected={selected === sat.name}
            onClick={() => onSelect(sat.name === selected ? null : sat.name)}
          />
        ))}

        {/* Retired header & sats */}
        {retiredSats.length > 0 && filter !== 'ACTIVE' && (
          <>
            <div style={{
              padding: '8px 12px 2px',
              fontSize: '9px',
              color: 'var(--text-dim)',
              letterSpacing: '2px',
              textTransform: 'uppercase',
            }}>
              Retired ({retiredSats.length})
            </div>
            {retiredSats.map(sat => (
              <SatCard
                key={sat.name}
                sat={sat}
                isSelected={selected === sat.name}
                onClick={() => onSelect(sat.name === selected ? null : sat.name)}
              />
            ))}
          </>
        )}

        {!loading && filteredSats.length === 0 && (
          <div style={{ textAlign: 'center', padding: '24px 12px', fontSize: '11px', color: 'var(--text-dim)' }}>
            No satellites match filter
          </div>
        )}
      </div>
    </div>
  )
}
