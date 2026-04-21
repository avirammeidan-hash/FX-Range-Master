import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import type { CorrelatedPair } from '@trading/api-client'

interface Props {
  pairs: CorrelatedPair[]
}

export default function CorrelationMatrix({ pairs }: Props) {
  if (!pairs || pairs.length === 0) {
    return (
      <div className="card">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Correlated Markets</h3>
        <p className="text-sm text-gray-600 text-center py-4">No correlation data available</p>
      </div>
    )
  }

  // Sort: biggest movers first
  const sorted = [...pairs].sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct))

  return (
    <div className="card">
      <h3 className="text-sm font-medium text-gray-400 mb-3">Correlated Markets</h3>
      <div className="grid grid-cols-2 gap-1.5 max-h-72 overflow-y-auto no-scrollbar">
        {sorted.map((p) => {
          const up   = p.change_pct > 0
          const down = p.change_pct < 0
          const Icon = up ? TrendingUp : down ? TrendingDown : Minus
          const color = up ? 'text-gain' : down ? 'text-loss' : 'text-gray-500'
          const barWidth = Math.min(100, Math.abs(p.change_pct) * 20)

          return (
            <div key={p.pair} className="bg-bg-700 rounded-lg px-2.5 py-2 relative overflow-hidden">
              {/* Change bar background */}
              <div
                className={`absolute inset-y-0 left-0 opacity-10 ${up ? 'bg-gain' : down ? 'bg-loss' : 'bg-gray-600'}`}
                style={{ width: `${barWidth}%` }}
              />
              <div className="relative flex items-center justify-between gap-2">
                <span className="text-xs text-gray-400 font-medium truncate">
                  {p.pair.replace('=X', '').replace('USD', '').replace('ILS', '').replace('EUR', 'EUR/') || p.pair}
                </span>
                <div className={`flex items-center gap-0.5 text-xs font-medium num ${color}`}>
                  <Icon className="w-3 h-3" />
                  {p.change_pct >= 0 ? '+' : ''}{p.change_pct.toFixed(2)}%
                </div>
              </div>
              {p.price != null && (
                <p className="relative text-xs num text-gray-500 mt-0.5">{p.price.toFixed(4)}</p>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
