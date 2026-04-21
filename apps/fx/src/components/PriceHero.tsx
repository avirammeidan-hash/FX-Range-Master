import { TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react'
import { FxStatus } from '../services/api'

interface Props {
  data: FxStatus
}

export default function PriceHero({ data }: Props) {
  const chg = data.daily_change_pct
  const isUp = chg > 0
  const isDown = chg < 0

  const Icon = isUp ? TrendingUp : isDown ? TrendingDown : Minus
  const chgColor = isUp ? 'text-gain' : isDown ? 'text-loss' : 'text-gray-400'

  // Position within the band (0=lower, 0.5=baseline, 1=upper)
  const range = data.upper - data.lower
  const pos = range > 0 ? (data.price - data.lower) / range : 0.5
  const posPct = Math.round(Math.max(0, Math.min(1, pos)) * 100)

  // Signal badge
  const sig = data.signal?.signal ?? 'FLAT'
  const sigStyle =
    sig === 'BUY'  ? 'bg-gain/20 text-gain border-gain/40' :
    sig === 'SELL' ? 'bg-loss/20 text-loss border-loss/40' :
    sig === 'HOLD' ? 'bg-warn/20 text-warn border-warn/40' :
    'bg-bg-700 text-gray-400 border-bg-600'

  return (
    <div className="card">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        {/* Price */}
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{data.pair}</p>
          <div className="flex items-baseline gap-3">
            <span className="text-5xl font-bold num text-white">{data.price.toFixed(4)}</span>
            <span className={`flex items-center gap-1 text-lg font-semibold num ${chgColor}`}>
              <Icon className="w-5 h-5" />
              {chg >= 0 ? '+' : ''}{chg.toFixed(3)}%
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Baseline <span className="num text-gray-300">{data.baseline.toFixed(4)}</span>
            {data.data_stale && (
              <span className="ml-2 text-warn flex items-center gap-1 inline-flex">
                <AlertTriangle className="w-3 h-3" /> stale
              </span>
            )}
          </p>
        </div>

        {/* Signal + Recommendation */}
        <div className="flex flex-col items-end gap-2">
          <span className={`px-4 py-1.5 rounded-full text-sm font-bold border ${sigStyle}`}>
            {sig}
          </span>
          <span className={`text-xs font-medium ${
            data.trade_recommendation === 'TRADE' ? 'text-gain' :
            data.trade_recommendation === 'SKIP'  ? 'text-loss' : 'text-warn'
          }`}>
            ML says: {data.trade_recommendation}
          </span>
        </div>
      </div>

      {/* ── Band position bar ──────────────────────────────────────────── */}
      <div className="mt-4">
        <div className="flex justify-between text-xs text-gray-500 mb-1 num">
          <span>Stop {data.stop_lower.toFixed(4)}</span>
          <span>Lower {data.lower.toFixed(4)}</span>
          <span>Base {data.baseline.toFixed(4)}</span>
          <span>Upper {data.upper.toFixed(4)}</span>
          <span>Stop {data.stop_upper.toFixed(4)}</span>
        </div>
        <div className="relative h-3 rounded-full bg-bg-700 overflow-visible">
          {/* Stop zones (red) */}
          <div className="absolute left-0 top-0 h-full w-[8%] rounded-l-full bg-loss/25" />
          <div className="absolute right-0 top-0 h-full w-[8%] rounded-r-full bg-loss/25" />
          {/* Window (blue) */}
          <div className="absolute left-[8%] right-[8%] top-0 h-full bg-accent/15" />
          {/* Baseline tick */}
          <div className="absolute left-1/2 top-0 h-full w-px bg-gray-600" />
          {/* Price marker */}
          <div
            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 rounded-full border-2 border-white bg-accent shadow-lg transition-all duration-500"
            style={{ left: `${posPct}%` }}
          />
        </div>
        <div className="flex justify-between text-xs mt-1">
          <span className="text-loss">−{(data.params.half_width_pct + data.params.stop_ext_pct).toFixed(1)}%</span>
          <span className="text-gray-500">{posPct}% in band</span>
          <span className="text-loss">+{(data.params.half_width_pct + data.params.stop_ext_pct).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  )
}
