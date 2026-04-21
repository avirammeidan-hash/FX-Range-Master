import { Star, TrendingUp, TrendingDown } from 'lucide-react'
import { usePolling } from '@trading/hooks'
import { getWatchlist, getQuote } from '../services/api'
import { useNavigate } from 'react-router-dom'

function WatchlistRow({ ticker }: { ticker: string }) {
  const { data: quote } = usePolling(['quote', ticker], () => getQuote(ticker), 30_000)
  const navigate = useNavigate()

  if (!quote) return (
    <div className="flex items-center justify-between px-4 py-3 bg-bg-700 rounded-lg animate-pulse">
      <div className="h-4 bg-bg-600 rounded w-16" />
      <div className="h-4 bg-bg-600 rounded w-20" />
    </div>
  )

  const isUp = quote.change_pct >= 0
  return (
    <button
      onClick={() => navigate(`/stocks?t=${ticker}`)}
      className="w-full flex items-center justify-between px-4 py-3 bg-bg-700 hover:bg-bg-600 rounded-lg transition-colors text-left"
    >
      <div>
        <p className="font-bold text-white">{quote.ticker}</p>
        <p className="text-xs text-gray-500 truncate max-w-[140px]">{quote.name}</p>
      </div>
      <div className="text-right">
        <p className="font-bold num">${quote.price.toFixed(2)}</p>
        <p className={`text-xs num flex items-center gap-1 justify-end ${isUp ? 'text-gain' : 'text-loss'}`}>
          {isUp ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
          {isUp ? '+' : ''}{quote.change_pct.toFixed(2)}%
        </p>
      </div>
    </button>
  )
}

export default function Watchlist() {
  const { data } = usePolling('watchlist', getWatchlist, 60_000)
  const tickers  = data?.tickers ?? []

  return (
    <div className="space-y-4 max-w-lg">
      <div className="flex items-center gap-2">
        <Star className="w-5 h-5 text-warn" />
        <h1 className="text-xl font-bold">Watchlist</h1>
        <span className="text-sm text-gray-500 ml-1">({tickers.length})</span>
      </div>

      {tickers.length === 0 ? (
        <div className="card text-center py-12 text-gray-500">
          <Star className="w-8 h-8 mx-auto mb-2 text-gray-700" />
          <p>No tickers yet — add from the Dashboard</p>
        </div>
      ) : (
        <div className="space-y-2">
          {tickers.map(t => <WatchlistRow key={t} ticker={t} />)}
        </div>
      )}
    </div>
  )
}
