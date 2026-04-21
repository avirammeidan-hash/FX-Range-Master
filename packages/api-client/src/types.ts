/**
 * Shared types used by both FX and Stocks apps.
 * App-specific types live in each app's own services/api.ts.
 */

// ── Candles (OHLCV) — universal for any asset ────────────────────────────────
export interface Candle {
  t: string   // ISO timestamp or YYYY-MM-DD
  o: number
  h: number
  l: number
  c: number
  v?: number
}

// ── Technical indicators — standard across asset classes ─────────────────────
export interface TechnicalIndicators {
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
  macd?: number
  macd_signal?: number
  macd_hist?: number
  volume_ratio?: number   // current vol / avg vol
  correlated_pairs?: CorrelatedPair[]
}

// ── Correlated assets ────────────────────────────────────────────────────────
export interface CorrelatedPair {
  pair: string
  price: number
  change_pct: number
  correlation?: number
}

// ── News & sentiment ─────────────────────────────────────────────────────────
export interface NewsArticle {
  title: string
  source: string
  url?: string
  timestamp?: string
  sentiment?: string        // BULLISH | BEARISH | NEUTRAL
  keywords?: string[]
  impact?: number           // 0–1
  summary?: string
}

export interface SentimentSummary {
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' | string
  score: number             // negative = bearish, positive = bullish
  alert_count: number
}

// ── Economic / earnings events ───────────────────────────────────────────────
export interface MarketEvent {
  date: string
  name: string
  impact: 'HIGH' | 'MEDIUM' | 'LOW' | string
  country?: string
  previous?: string
  forecast?: string
  actual?: string
  description?: string
  ticker?: string           // for earnings events
}

// ── Price line — for CandleChart overlays ────────────────────────────────────
export interface PriceLine {
  price: number
  color: string
  title: string
  lineStyle?: 0 | 1 | 2 | 3   // 0=solid, 1=dotted, 2=dashed, 3=large-dashed
  lineWidth?: 1 | 2 | 3
}

// ── ML / AI confidence — generic gauge data ──────────────────────────────────
export interface ConfidenceResult {
  confidence: number        // 0–1
  decision: string          // e.g. 'TRADE' | 'SKIP' | 'BUY' | 'SELL'
  reason: string
  available: boolean
  top_feature?: string
  top_feature_value?: number
  model_accuracy?: number
}
