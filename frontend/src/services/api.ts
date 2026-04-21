// FX-Range-Master API service
// Connects to Flask backend at localhost:5000 (proxied via Vite /api)

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`API ${res.status}: ${url}`)
  return res.json()
}

async function postJSON<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${url}`)
  return res.json()
}

// ── Main status ───────────────────────────────────────────────────────────────
export const getStatus = () => fetchJSON<FxStatus>('/api/data')
export const resetBaseline = () => fetchJSON<{ ok: boolean }>('/api/reset')

// ── Candles ───────────────────────────────────────────────────────────────────
export const getCandles = () => fetchJSON<CandleResponse>('/api/candles')

// ── News ──────────────────────────────────────────────────────────────────────
export const getNews = () => fetchJSON<NewsResponse>('/api/news')
export const refreshNews = () => fetchJSON<NewsResponse>('/api/news/refresh')

// ── Economic calendar ─────────────────────────────────────────────────────────
export const getCalendar = () => fetchJSON<CalendarResponse>('/api/econ-calendar')

// ── AI Performance ────────────────────────────────────────────────────────────
export const getAiPerformance = () => fetchJSON<AiPerfResponse>('/api/ai-performance')

// ── ML Export ────────────────────────────────────────────────────────────────
export const getMlExport = () => fetchJSON<MlExportResponse>('/api/ml-export')

// ── Retrain ───────────────────────────────────────────────────────────────────
export const retrain = () => postJSON<{ ok: boolean; message?: string }>('/api/retrain')

// ── Types ─────────────────────────────────────────────────────────────────────

export interface FxStatus {
  pair: string
  price: number
  baseline: number
  upper: number
  lower: number
  stop_upper: number
  stop_lower: number
  daily_change_pct: number
  dist_upper_pct: number
  dist_lower_pct: number
  position: 'LONG' | 'SHORT' | 'FLAT'
  signal: Signal
  signals_history: Signal[]
  params: { half_width_pct: number; stop_ext_pct: number }
  today_events: CalendarEvent[]
  trade_recommendation: 'TRADE' | 'SKIP' | 'CAUTION'
  vix: number | null
  news_alerts: NewsAlert[]
  news_sentiment: NewsSentiment
  ml_prediction: MlPrediction | null
  technical: TechnicalData
  data_source: string
  data_stale: boolean
  error?: string
}

export interface Signal {
  signal: 'BUY' | 'SELL' | 'HOLD' | 'FLAT' | string
  price?: number
  reason?: string
  timestamp?: string
  context?: string
}

export interface MlPrediction {
  trade: boolean
  confidence: number
  raw_confidence: number
  threshold: number
  prediction: number
  reason: string
  ml_available: boolean
  features?: Record<string, number>
  top_feature?: string
  top_feature_value?: number
  model_accuracy?: number
  model_age_days?: number
}

export interface TechnicalData {
  available: boolean
  ema20?: number
  ema50?: number
  ema_cross?: 'golden' | 'death' | 'neutral'
  rsi14?: number
  bb_upper?: number
  bb_lower?: number
  bb_mid?: number
  atr?: number
  momentum?: number
  trend?: 'UP' | 'DOWN' | 'SIDEWAYS'
  correlated_pairs?: CorrelatedPair[]
}

export interface CorrelatedPair {
  pair: string
  price: number
  change_pct: number
  correlation?: number
}

export interface NewsSentiment {
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' | string
  score: number
  alert_count: number
}

export interface NewsAlert {
  title: string
  source: string
  url?: string
  timestamp?: string
  sentiment?: string
  keywords?: string[]
  impact?: number
}

export interface CalendarEvent {
  date: string
  name: string
  impact: 'HIGH' | 'MEDIUM' | 'LOW' | string
  country?: string
  previous?: string
  forecast?: string
  actual?: string
  description?: string
}

export interface CandleResponse {
  candles: Candle[]
  pair: string
  interval?: string
}

export interface Candle {
  t: string    // ISO timestamp
  o: number
  h: number
  l: number
  c: number
  v?: number
}

export interface NewsResponse {
  alerts: NewsAlert[]
  sentiment: NewsSentiment
  new_count?: number
}

export interface CalendarResponse {
  events: CalendarEvent[]
  technical?: TechnicalData
  macro?: Record<string, unknown>
}

export interface AiPerfResponse {
  summary: {
    total: number
    correct: number
    accuracy_pct: number
    trade_accuracy_pct: number
    skip_accuracy_pct: number
    period_days: number
  }
  recent: AiDecision[]
  error?: string
}

export interface AiDecision {
  timestamp: string
  ml_decision: 'TRADE' | 'SKIP'
  confidence: number
  correct: boolean | null
  price_at: number
  price_after: number
  change_pct: number
}

export interface MlExportResponse {
  status: 'trained' | 'not_trained' | string
  train_date?: string
  accuracy?: number
  threshold?: number
  feature_importance?: Record<string, number>
  training_days?: number
  records?: unknown[]
}
