// Stocks app API — connects to Flask backend at localhost:5001
import { fetchJSON, postJSON } from '@trading/api-client'
export type {
  Candle,
  TechnicalIndicators,
  NewsArticle,
  SentimentSummary,
  MarketEvent,
} from '@trading/api-client'

// ── Endpoint calls ────────────────────────────────────────────────────────────
export const getQuote       = (ticker: string) => fetchJSON<StockQuote>(`/api/quote/${ticker}`)
export const getCandles     = (ticker: string, interval = '1d') =>
  fetchJSON<CandlesResponse>(`/api/candles/${ticker}?interval=${interval}`)
export const getTechnicals  = (ticker: string) => fetchJSON<TechResponse>(`/api/technicals/${ticker}`)
export const getNews        = (ticker?: string) =>
  fetchJSON<NewsResponse>(ticker ? `/api/news/${ticker}` : '/api/news')
export const getCalendar    = (ticker: string) => fetchJSON<CalendarResponse>(`/api/calendar/${ticker}`)
export const getSentiment   = () => fetchJSON<MarketSentiment>('/api/sentiment')
export const getWatchlist   = () => fetchJSON<WatchlistResponse>('/api/watchlist')
export const addToWatchlist = (ticker: string) => postJSON<{ ok: boolean }>('/api/watchlist', { ticker })

// ── Stock-specific types ──────────────────────────────────────────────────────
import type { Candle, TechnicalIndicators, NewsArticle, SentimentSummary, MarketEvent } from '@trading/api-client'

export interface StockQuote {
  ticker: string
  name: string
  price: number
  open: number
  high: number
  low: number
  prev_close: number
  change: number
  change_pct: number
  volume: number
  avg_volume: number
  market_cap?: number
  pe_ratio?: number
  week_52_high?: number
  week_52_low?: number
  sector?: string
  industry?: string
  currency: string
  exchange: string
  timestamp: string
}

export interface CandlesResponse {
  ticker: string
  interval: string
  candles: Candle[]
}

export interface TechResponse {
  ticker: string
  indicators: TechnicalIndicators
}

export interface NewsResponse {
  articles: NewsArticle[]
  sentiment: SentimentSummary
}

export interface CalendarResponse {
  ticker: string
  events: MarketEvent[]   // earnings, dividends, splits
}

export interface MarketSentiment {
  fear_greed_index: number          // 0–100 (0 = extreme fear)
  fear_greed_label: string          // 'Extreme Fear' | 'Fear' | 'Neutral' | 'Greed' | 'Extreme Greed'
  fear_greed_history: { date: string; value: number }[]
  sector_performance: SectorPerf[]
}

export interface SectorPerf {
  sector: string
  change_pct: number
}

export interface WatchlistResponse {
  tickers: string[]
}
