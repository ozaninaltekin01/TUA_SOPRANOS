/**
 * App.jsx — Ana layout
 *
 * Layout:
 *   StatusBar (top)
 *   ├── SatellitePanel (left, 240px)
 *   ├── Globe          (center, flex)
 *   └── ThreatPanel    (right, 280px — only when sat selected)
 *   BottomBar (bottom)
 */

import { useState, useCallback } from 'react'
import { useApiData }      from './hooks/useApiData'
import Globe               from './components/Globe'
import SatellitePanel      from './components/SatellitePanel'
import ThreatPanel         from './components/ThreatPanel'
import StatusBar           from './components/StatusBar'

// ── Legend ────────────────────────────────────────────────────
function Legend() {
  return (
    <div className="legend">
      <div className="legend-item">
        <div className="legend-line sgp4" />
        SGP4 Orbit
      </div>
      <div className="legend-item">
        <div className="legend-line lstm" />
        LSTM Predicted
      </div>
      <div className="legend-item">
        <div className="legend-line red" />
        RED Threat
      </div>
      <div className="legend-item">
        <div className="legend-line yellow" />
        YELLOW Threat
      </div>
      <div className="legend-item">
        <div className="legend-line watch" />
        WATCH
      </div>
      <div className="legend-item" style={{ marginLeft: 8 }}>
        <div className="legend-dot" style={{ background: 'var(--cyan)' }} />
        Primary Sat
      </div>
      <div className="legend-item">
        <div className="legend-dot" style={{ background: 'var(--red)' }} />
        Threat (TCA pos)
      </div>
    </div>
  )
}

// ── Demo Toggle ───────────────────────────────────────────────
function DemoToggle({ value, onChange }) {
  return (
    <div className="demo-toggle-wrap">
      <span className="demo-label">DEMO</span>
      <label className="toggle-switch">
        <input
          type="checkbox"
          checked={value}
          onChange={e => onChange(e.target.checked)}
        />
        <span className="toggle-slider" />
      </label>
      {value && <span className="demo-active-text">● ON</span>}
    </div>
  )
}

// ── Loading overlay ───────────────────────────────────────────
function LoadingOverlay({ visible }) {
  if (!visible) return null
  return (
    <div className="loading-overlay">
      <div className="loading-spinner" />
      <div className="loading-text">INITIALIZING ORBITAL DATA</div>
      <div className="loading-sub">Connecting to backend · Loading TLEs · Running SGP4</div>
    </div>
  )
}

// ── Globe hint ────────────────────────────────────────────────
function GlobeHint({ visible }) {
  if (!visible) return null
  return (
    <div className="globe-hint">
      ← Select a satellite to begin analysis
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────
export default function App() {
  const [selectedSatName, setSelectedSatName] = useState(null)
  const [demoMode,        setDemoMode]        = useState(false)
  const [globeReady,      setGlobeReady]      = useState(false)

  const { data, loading, error, lastFetch, apiOnline, refresh } = useApiData()

  const satellites   = data?.satellites || []
  const selectedSat  = satellites.find(s => s.name === selectedSatName) || null

  const handleSelect = useCallback((name) => {
    setSelectedSatName(name)
  }, [])

  const handleGlobeReady = useCallback(() => {
    setGlobeReady(true)
  }, [])

  // API offline → auto-enable demo mode with an alert
  const showOfflineWarning = !loading && !apiOnline && !demoMode

  return (
    <div className="app">
      <StatusBar
        data={data}
        loading={loading}
        apiOnline={apiOnline}
        lastFetch={lastFetch}
      />

      {showOfflineWarning && (
        <div style={{
          background: 'rgba(255,136,0,0.12)',
          borderBottom: '1px solid rgba(255,136,0,0.3)',
          padding: '6px 20px',
          fontSize: '11px',
          color: 'var(--orange)',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          flexShrink: 0,
        }}>
          ⚠️ API offline — make sure FastAPI is running on port 8000.
          <span style={{ color: 'var(--text-dim)' }}>
            cd tua_sopranos1 &amp;&amp; python -m uvicorn api:app --reload --port 8000
          </span>
          <button
            onClick={refresh}
            style={{
              marginLeft: 'auto',
              background: 'transparent',
              border: '1px solid var(--orange)',
              color: 'var(--orange)',
              padding: '2px 10px',
              borderRadius: 4,
              cursor: 'pointer',
              fontSize: '10px',
            }}
          >
            RETRY
          </button>
        </div>
      )}

      <div className="main-area">
        {/* Left — satellite list */}
        <SatellitePanel
          satellites={satellites}
          selected={selectedSatName}
          onSelect={handleSelect}
          loading={loading}
        />

        {/* Center — globe */}
        <div className="globe-container">
          <Globe
            allSatellites={satellites}
            selectedSat={selectedSat}
            demoMode={demoMode}
            onGlobeReady={handleGlobeReady}
          />
          <LoadingOverlay visible={loading && !globeReady} />
          <GlobeHint visible={globeReady && !selectedSatName} />
        </div>

        {/* Right — threat panel (only when sat selected) */}
        {selectedSat && (
          <ThreatPanel
            satellite={selectedSat}
            demoMode={demoMode}
          />
        )}
      </div>

      {/* Bottom bar */}
      <div className="bottom-bar">
        <Legend />
        <div className="status-bar__divider" />
        <DemoToggle value={demoMode} onChange={setDemoMode} />
      </div>
    </div>
  )
}
