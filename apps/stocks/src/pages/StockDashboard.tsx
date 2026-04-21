import { useState } from 'react'
import { Search, TrendingUp, TrendingDown, RefreshCw, Plus } from 'lucide-react'
import { usePolling } from '@trading/hooks'
import { CandleChart, TechPanel, NewsList, AccuracyGauge } from '@trading/ui'
import { getQuote, getCandles, getTechnicals, getNews, getSentiment, addToWatchlist } from '../services/api'

const DEFAULT_TICKER = 'AAPL'

export default function StockDashboard() {
  const [ticker, setTicker]   = useState(DEFAULT_TICKER)
  const [input,  setInput]    = useState(DEFAULT_TICKER)
  const [adding, setAdding]   = useState(false)

  const { data: quote,  isLoading: qLoading  } = usePolling(['quote',      ticker], () => getQuote(ticker),      30_000)
  const { data: candles                       } = usePolling(['candles',    ticker], () => getCandles(ticker),    60_000)
  const { data: tech                          } = usePolling(['technicals', ticker], () => getTechnicals(ticker), 60_000)
  const { data: news                          } = usePolling(['news',       ticker], () => getNews(ticker),       60_000)
  const { data: sentiment                     } = usePolling('sentiment',            getSentiment,                300_000)

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim()) setTicker(input.trim().toUpperCase())
  }

  const handleAddWatchlist = async () => {
    setAdding(true)
    await addToWatchlist(ticker).catch(() => {})
    setAdding(false)
  }

  const changeColor = (quote?.change_pct ?? 0) >= 0 ? 'text-gain' : 'text-loss'
  const ChangeIcon  = (quote?.change_pct ?? 0) >= 0 ? TrendingUp  : TrendingDown

  return (
    <div className="space-y-6">
      {/* Search bar */}
      <div className="flex items-center gap-3">
        <form onSubmit={handleSearch} className="flex gap-2 flex-1 max-w-sm">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
            <input
              value={input}
              onChange={e => setInput(e.target.value.toUpperCase())}
              placeholder="AAPL, TSLA, NVDA…"
              className="w-full pl-9 pr-3 py-2 bg-bg-700 border border-bg-600 rounded-lg text-sm
                         text-gray-200 placeholder-gray-600 focus:outline-none focus:border-accent"
            />
          </div>
          <button type="submit" className="btn-primary">Search</button>
        </form>
        <button
          onClick={handleAddWatchlist}
          disabled={adding}
          className="btn-ghost flex items-center gap-1.5 text-xs"
        >
          <Plus className="w-3.5 h-3.5" /> Watchlist
        </button>
      </div>

      {qLoading ? (
        <div className="card flex items-center justify-center py-16 text-gray-500">
          <RefreshCw className="w-5 h-5 animate-spin mr-2" /> Loading {ticker}…
        </div>
      ) : quote ? (
        <>
          {/* Price hero */}
          <div className="card">
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-baseline gap-3">
                  <h1 className="text-4xl font-bold num text-white">{quote.ticker}</h1>
                  <span className="text-gray-500 text-sm">{quote.name}</span>
                </div>
                <div className="flex items-baseline gap-3 mt-2">
                  <span className="text-3xl font-bold num">
                    {quote.currency === 'USD' ? '$' : ''}{quote.price.toFixed(2)}
                  </span>
                  <span className={`flex items-center gap-1 font-semibold ${changeColor}`}>
                    <ChangeIcon className="w-4 h-4" />
                    {quote.change >= 0 ? '+' : ''}{quote.change.toFixed(2)}
                    ({quote.change_pct >= 0 ? '+' : ''}{quote.change_pct.toFixed(2)}%)
                  </span>
                </div>
              </div>
              <div className="text-right space-y-1">
                {quote.market_cap && (
                  <div className="text-xs text-gray-500">
                    Cap <span className="text-gray-300 font-medium">
                      ${(quote.market_cap / 1e9).toFixed(1)}B
                    </span>
                  </div>
                )}
                {quote.pe_ratio && (
                  <div className="text-xs text-gray-500">
                    P/E <span className="text-gray-300 font-medium">{quote.pe_ratio.toFixed(1)}</span>
                  </div>
                )}
                <div className="text-xs text-gray-500">
                  {quote.exchange} · {quote.sector ?? '—'}
                </div>
              </div>
            </div>

            {/* 52w range */}
            {quote.week_52_low != null && quote.week_52_high != null && (
              <div className="mt-4">
                <div className="flex justify-between text-xs text-gray-500 mb-1">
                  <span>52w Low ${quote.week_52_low.toFixed(2)}</span>
                  <span>52w High ${quote.week_52_high.toFixed(2)}</span>
                </div>
                <div className="relative h-1.5 bg-bg-600 rounded-full">
                  <div
                    className="absolute left-0 top-0 h-full bg-accent rounded-full"
                    style={{
                      width: `${((quote.price - quote.week_52_low) / (quote.week_52_high - quote.week_52_low)) * 100}%`,
                    }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Chart + Tech */}
          <div className="grid grid-cols-3 gap-4">
            <div className="col-span-2 card">
              <h3 className="text-sm font-medium text-gray-400 mb-3">Price Chart</h3>
              <CandleChart candles={candles?.candles ?? []} height={320} />
            </div>
            <div>
              {tech ? (
                <TechPanel tech={tech.indicators} title={`${ticker} Technicals`} />
              ) : (
                <div className="card text-sm text-gray-500 text-center py-8">Loading TA…</div>
              )}
            </div>
          </div>

          {/* Sentiment + News */}
          <div className="grid grid-cols-3 gap-4">
            {sentiment && (
              <div className="card flex flex-col items-center">
                <h3 className="text-sm font-medium text-gray-400 mb-3 w-full">Fear & Greed</h3>
                <AccuracyGauge
                  pct={sentiment.fear_greed_index}
                  label={`${sentiment.fear_greed_index}`}
                  sublabel={sentiment.fear_greed_label}
                  threshold={50}
                />
                <div className="mt-4 w-full space-y-1">
                  {sentiment.sector_performance.slice(0, 5).map(s => (
                    <div key={s.sector} className="flex justify-between text-xs">
                      <span className="text-gray-500 truncate">{s.sector}</span>
                      <span className={s.change_pct >= 0 ? 'text-gain num' : 'text-loss num'}>
                        {s.change_pct >= 0 ? '+' : ''}{s.change_pct.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="col-span-2">
              <NewsList
                articles={news?.articles ?? []}
                sentiment={news?.sentiment ?? { sentiment: 'NEUTRAL', score: 0, alert_count: 0 }}
                maxHeight={320}
              />
            </div>
          </div>
        </>
      ) : (
        <div className="card text-center py-16 text-gray-500">
          Search for a ticker to get started
        </div>
      )}
    </div>
  )
}
