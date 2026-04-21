// FX-Range-Master API — USD/ILS specific endpoints
// Base fetch helpers + shared types come from @trading/api-client
import { fetchJSON, postJSON } from '@trading/api-client'
import type {
  Candle,
  TechnicalIndicators,
  NewsArticle,
  SentimentSummary,
  MarketEvent,
} from '@trading/api-client'

// Re-export shared types under FX-familiar names for backward compat
export type { Candle, CorrelatedPair } from '@trading/api-client'
export type TechnicalData   = TechnicalIndicators
export type NewsAlert        = NewsArticle
export type NewsSentiment    = SentimentSummary
export type CalendarEvent    = MarketEvent

// ── Endpoint calls ────────────────────────────────────────────────────────────
export const getStatus        = () => fetchJSON<FxStatus>('/api/data')
export const resetBaseline    = () => fetchJSON<{ ok: boolean }>('/api/reset')
export const getCandles       = () => fetchJSON<CandleResponse>('/api/candles')
export const getNews          = () => fetchJSON<NewsResponse>('/api/news')
export const refreshNews      = () => fetchJSON<NewsResponse>('/api/news/refresh')
export const getCalendar      = () => fetchJSON<CalendarResponse>('/api/econ-calendar')
export const getAiPerformance = () => fetchJSON<AiPerfResponse>('/api/ai-performance')
export const getMlExport      = () => fetchJSON<MlExportResponse>('/api/ml-export')
export const retrain              = () => postJSON<{ ok: boolean; message?: string }>('/api/retrain')
export const getSignalPerformance = () => fetchJSON<SignalPerformanceResponse>('/api/signal-performance')

// ── FX-specific types ─────────────────────────────────────────────────────────
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
  params: { half_width_pct: number; stop_ext_pct: number; stop_adaptive?: boolean }
  today_events: MarketEvent[]
  trade_recommendation: 'TRADE' | 'SKIP' | 'CAUTION'
  vix: number | null
  news_alerts: NewsArticle[]
  news_sentiment: SentimentSummary
  ml_prediction: MlPrediction | null
  technical: TechnicalIndicators
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

export interface CandleResponse {
  candles: Candle[]
  pair: string
  interval?: string
}

export interface NewsResponse {
  alerts: NewsArticle[]
  sentiment: SentimentSummary
  new_count?: number
}

export interface CalendarResponse {
  events: MarketEvent[]
  technical?: TechnicalIndicators
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
  prediction_time?: string
  lookback_min?: number
  price_at_prediction?: number
  price_now?: number
  price_change_pct?: number
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

// ── Signal Performance (deep research) ───────────────────────────────────────
export interface SignalPerformanceResponse {
  calibration: CalibrationBucket[]
  accuracy_by_day: DayAccuracy[]
  pnl_curve: PnlPoint[]
  pnl_summary: PnlSummary
  signal_breakdown: Record<string, SignalBreakdown>
  hints: Hint[]
  total_records: number
  error?: string
}

export interface CalibrationBucket {
  bucket: string       // e.g. "50–65%"
  expected_pct: number // avg model confidence in bucket
  actual_pct: number   // actual % correct
  count: number
  gap: number          // actual - expected (positive = better than expected)
}

export interface DayAccuracy {
  date: string
  accuracy: number
  total: number
}

export interface PnlPoint {
  date: string
  equity: number
}

export interface PnlSummary {
  start_equity: number
  end_equity: number
  return_pct: number
  win_trades: number
  lose_trades: number
  total_trades: number
  win_rate: number
}

export interface SignalBreakdown {
  total: number
  wins: number
  losses: number
  accuracy: number
}

export interface Hint {
  type: string
  message: string
  severity: 'high' | 'medium' | 'low'
}
