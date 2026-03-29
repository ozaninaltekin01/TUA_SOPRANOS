/**
 * useApiData.js — FastAPI'den veri cekme hook'u
 *
 * Her 10 dakikada bir veriyi yeniler (API onbellegiyle uyumlu).
 * API'ye ulasalamazsa demoMode otomatik devreye girer.
 */

import { useState, useEffect, useCallback, useRef } from 'react'

const API_URL  = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const POLL_MS  = 10 * 60 * 1000  // 10 dakika

export function useApiData() {
  const [data,      setData]      = useState(null)
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState(null)
  const [lastFetch, setLastFetch] = useState(null)
  const [apiOnline, setApiOnline] = useState(true)
  const timerRef = useRef(null)

  const fetchData = useCallback(async () => {
    try {
      setError(null)
      const res = await fetch(`${API_URL}/api/full`, {
        signal: AbortSignal.timeout(30_000), // 30s timeout
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const json = await res.json()
      setData(json)
      setLastFetch(new Date())
      setApiOnline(true)
    } catch (err) {
      console.warn('[useApiData] API error:', err.message)
      setError(err.message)
      setApiOnline(false)
    } finally {
      setLoading(false)
    }
  }, [])

  // Ilk yuklemede ve sonra her POLL_MS'de bir cek
  useEffect(() => {
    fetchData()
    timerRef.current = setInterval(fetchData, POLL_MS)
    return () => clearInterval(timerRef.current)
  }, [fetchData])

  const refresh = useCallback(() => {
    setLoading(true)
    fetchData()
  }, [fetchData])

  return { data, loading, error, lastFetch, apiOnline, refresh }
}

/**
 * useSystemStatus — hafif durum endpoint'i (sadece ozet, yol verisi yok)
 */
export function useSystemStatus() {
  const [status,  setStatus]  = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    fetch(`${API_URL}/api/status`, { signal: AbortSignal.timeout(5_000) })
      .then(r => r.ok ? r.json() : null)
      .then(json => { if (!cancelled && json) setStatus(json) })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [])

  return { status, loading }
}
