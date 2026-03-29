/**
 * StatusBar.jsx — Ust bilgi cubugu
 * Logo | LIVE badge + son guncelleme | Uyari sayaclar | Sistem durumu
 */

export default function StatusBar({ data, loading, apiOnline, lastFetch }) {
  const summary      = data?.summary       || {}
  const systemStatus = data?.system_status || (loading ? 'LOADING' : 'OFFLINE')

  const timeStr = lastFetch
    ? lastFetch.toUTCString().slice(17, 25) + ' UTC'
    : '—'

  return (
    <div className="status-bar">
      {/* Logo */}
      <div className="status-bar__logo">
        TUA <span>SOPRANOS</span>
      </div>

      <div className="status-bar__divider" />

      {/* Live badge */}
      <div className="live-badge">
        <div
          className={`live-dot ${
            loading    ? 'loading' :
            !apiOnline ? 'offline' : ''
          }`}
        />
        {loading    ? 'LOADING…'  :
         !apiOnline ? 'OFFLINE'   : 'LIVE'}
        &nbsp;
        <span style={{ color: 'var(--text-dim)', fontSize: '10px' }}>
          {timeStr}
        </span>
      </div>

      <div className="status-bar__divider" />

      {/* Alert counters */}
      <div className="alert-counters">
        <div className="counter-badge red">
          🔴 {summary.RED ?? 0} RED
        </div>
        <div className="counter-badge yellow">
          🟡 {summary.YELLOW ?? 0} YELLOW
        </div>
        <div className="counter-badge green">
          🟢 {summary.GREEN ?? 0} NOMINAL
        </div>
      </div>

      <div className="status-bar__spacer" />

      {/* System status chip */}
      {systemStatus !== 'LOADING' && systemStatus !== 'OFFLINE' && (
        <div className={`sys-status ${systemStatus}`}>
          {systemStatus}
        </div>
      )}
    </div>
  )
}
