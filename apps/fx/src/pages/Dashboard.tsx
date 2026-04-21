import { RefreshCw, AlertTriangle } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { usePolling } from '../hooks/usePolling'
import { getStatus, getNews } from '../services/api'
import PriceHero from '../components/PriceHero'
import MLConfidence from '../components/MLConfidence'
import SignalPanel from '../components/SignalPanel'
import { NewsList } from '../components/NewsCard'
import TechPanel from '../components/TechPanel'
import CorrelationMatrix from '../components/CorrelationMatrix'

export default function Dashboard() {
  const qc = useQueryClient()
  const { data, isLoading, isError, error } = usePolling('status', getStatus, 5000)
  const { data: newsData } = usePolling('news', getNews, 30000)

  const lastUpdated = new Date().toLocaleTimeString()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center space-y-3">
          <RefreshCw className="w-8 h-8 text-accent animate-spin mx-auto" />
          <p className="text-gray-400 text-sm">Connecting to FX-Range-Master...</p>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="card text-center max-w-md space-y-3">
          <AlertTriangle className="w-8 h-8 text-loss mx-auto" />
          <p className="text-loss font-medium">Cannot reach backend</p>
          <p className="text-xs text-gray-500">{String(error)}</p>
          <p className="text-xs text-gray-600">Make sure Flask is running on port 5000</p>
        </div>
      </div>
    )
  }

  if (!data) return null

  const tech = data.technical ?? { available: false }
  const pairs = tech.correlated_pairs ?? []

  return (
    <div className="space-y-4">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Dashboard</h1>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="live-dot" />
            <p className="text-xs text-gray-500">Live · updated {lastUpdated}</p>
          </div>
        </div>
        <button
          className="btn-ghost flex items-center gap-1.5 text-xs"
          onClick={() => qc.invalidateQueries({ queryKey: ['status'] })}
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh
        </button>
      </div>

      {/* ── Price Hero (full width) ──────────────────────────────────────── */}
      <PriceHero data={data} />

      {/* ── Row 1: Signal + ML ──────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <SignalPanel data={data} />
        </div>
        <MLConfidence ml={data.ml_prediction} />
      </div>

      {/* ── Row 2: Technical + Correlations ─────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <TechPanel tech={tech} />
        <CorrelationMatrix pairs={pairs} />
      </div>

      {/* ── Row 3: News ─────────────────────────────────────────────────── */}
      <NewsList
        articles={newsData?.alerts ?? data.news_alerts ?? []}
        sentiment={newsData?.sentiment ?? data.news_sentiment ?? { sentiment: 'NEUTRAL', score: 0, alert_count: 0 }}
      />

      {/* ── VIX badge ───────────────────────────────────────────────────── */}
      {data.vix != null && (
        <div className="card-sm flex items-center gap-3">
          <span className="text-xs text-gray-500">VIX</span>
          <span className={`num font-bold ${data.vix > 30 ? 'text-loss' : data.vix > 20 ? 'text-warn' : 'text-gain'}`}>
            {data.vix.toFixed(2)}
          </span>
          <span className="text-xs text-gray-500">
            {data.vix > 30 ? '⚠️ High volatility — extra caution' : data.vix > 20 ? 'Moderate volatility' : 'Low volatility'}
          </span>
        </div>
      )}
    </div>
  )
}
