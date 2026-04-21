import { useEffect, useCallback, useState } from 'react'

interface WSMessage {
  type: string
  data: unknown
  timestamp: string
}

// ── Module-level singleton — shared across ALL components ──────────────────
let globalWs: WebSocket | null = null
let globalConnected = false
const globalListeners = new Map<string, Set<(data: unknown) => void>>()
let connectionCount = 0
let reconnectTimer: ReturnType<typeof setTimeout> | null = null

function getWsUrl(path = '/ws'): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${path}`
}

function connectGlobal(path?: string) {
  if (globalWs && (globalWs.readyState === WebSocket.OPEN || globalWs.readyState === WebSocket.CONNECTING)) return

  globalWs = new WebSocket(getWsUrl(path))

  globalWs.onopen = () => {
    globalConnected = true
    globalWs?.send(JSON.stringify({ type: 'ping' }))
  }

  globalWs.onmessage = (event) => {
    try {
      const msg: WSMessage = JSON.parse(event.data)
      globalListeners.get(msg.type)?.forEach(cb => cb(msg.data))
    } catch { /* ignore parse errors */ }
  }

  globalWs.onclose = () => {
    globalConnected = false
    globalWs = null
    if (connectionCount > 0) {
      if (reconnectTimer) clearTimeout(reconnectTimer)
      reconnectTimer = setTimeout(() => connectGlobal(path), 3000)
    }
  }

  globalWs.onerror = () => { globalWs?.close() }
}

function disconnectGlobal() {
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null }
  if (globalWs) { globalWs.onclose = null; globalWs.close(); globalWs = null }
  globalConnected = false
}

/**
 * Global singleton WebSocket — one connection shared across all components.
 * Auto-connects on first subscriber, auto-reconnects on drop (3 s backoff).
 *
 * @param path  WebSocket URL path (default: '/ws')
 *
 * @example
 * const { connected, subscribe } = useWebSocket()
 * useEffect(() => subscribe('price_update', (data) => setPrice(data as number)), [subscribe])
 */
export function useWebSocket(path = '/ws') {
  const [connected, setConnected] = useState(globalConnected)

  useEffect(() => {
    connectionCount++
    connectGlobal(path)

    const interval = setInterval(() => setConnected(globalConnected), 1000)

    return () => {
      connectionCount--
      clearInterval(interval)
      if (connectionCount <= 0) { connectionCount = 0; disconnectGlobal() }
    }
  }, [path])

  const subscribe = useCallback((type: string, callback: (data: unknown) => void) => {
    if (!globalListeners.has(type)) globalListeners.set(type, new Set())
    globalListeners.get(type)!.add(callback)
    return () => { globalListeners.get(type)?.delete(callback) }
  }, [])

  const send = useCallback((type: string, data?: unknown) => {
    if (globalWs?.readyState === WebSocket.OPEN) {
      globalWs.send(JSON.stringify({ type, data, timestamp: new Date().toISOString() }))
    }
  }, [])

  return { connected, subscribe, send }
}
