import { Brain, TrendingUp, TrendingDown } from 'lucide-react'
import { MlPrediction } from '../services/api'

interface Props {
  ml: MlPrediction | null
}

export default function MLConfidence({ ml }: Props) {
  if (!ml?.ml_available) {
    return (
      <div className="card flex flex-col items-center py-6 text-center">
        <Brain className="w-8 h-8 text-gray-600 mb-2" />
        <p className="text-sm text-gray-500">ML Filter Unavailable</p>
        <p className="text-xs text-gray-600 mt-1">{ml?.reason ?? 'Model not loaded'}</p>
      </div>
    )
  }

  const pct = Math.round((ml.confidence ?? 0) * 100)
  const degrees = (pct / 100) * 180

  const color =
    pct >= 75 ? '#22c55e' :
    pct >= 55 ? '#f59e0b' :
    pct >= 40 ? '#f97316' : '#ef4444'

  const colorClass =
    pct >= 75 ? 'text-gain' :
    pct >= 55 ? 'text-warn' :
    pct >= 40 ? 'text-orange' : 'text-loss'

  const label =
    pct >= 75 ? 'Strong signal' :
    pct >= 55 ? 'Moderate signal' :
    pct >= 40 ? 'Weak signal' : 'No signal'

  return (
    <div className="card flex flex-col items-center">
      <div className="flex items-center gap-2 mb-3 w-full">
        <Brain className="w-4 h-4 text-accent" />
        <span className="text-xs text-gray-400 uppercase tracking-wider">ML Skip Filter</span>
        <span className={`ml-auto text-xs font-bold px-2 py-0.5 rounded-full ${
          ml.trade ? 'bg-gain/20 text-gain' : 'bg-loss/20 text-loss'
        }`}>
          {ml.trade ? 'TRADE' : 'SKIP'}
        </span>
      </div>

      {/* ── Gauge SVG ──────────────────────────────────────────────────── */}
      <div className="relative w-44 h-24 mb-1">
        <svg viewBox="0 0 200 110" className="w-full h-full">
          {/* BG arc */}
          <path d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none" stroke="#1e293b" strokeWidth="18" strokeLinecap="round" />
          {/* Value arc */}
          <path d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none" stroke={color} strokeWidth="18" strokeLinecap="round"
            strokeDasharray={`${(degrees / 180) * 251.2} 251.2`}
            style={{ transition: 'stroke-dasharray 0.8s ease' }}
          />
          {/* Threshold marker at 55% */}
          {(() => {
            const tAngle = (0.55 * Math.PI)
            const tx = 100 - 90 * Math.cos(tAngle)
            const ty = 100 - 90 * Math.sin(tAngle)
            const tx2 = 100 - 74 * Math.cos(tAngle)
            const ty2 = 100 - 74 * Math.sin(tAngle)
            return <line x1={tx} y1={ty} x2={tx2} y2={ty2} stroke="#4f8eff" strokeWidth="2" />
          })()}
          {/* Tick labels */}
          {[0, 25, 50, 75, 100].map(tick => {
            const angle = (tick / 100) * Math.PI
            const x = 100 - 102 * Math.cos(angle)
            const y = 100 - 102 * Math.sin(angle)
            return (
              <text key={tick} x={x} y={y} textAnchor="middle" dominantBaseline="middle"
                fill="#4b5563" fontSize="8">{tick}%</text>
            )
          })}
        </svg>
        {/* Center value */}
        <div className="absolute inset-0 flex flex-col items-center justify-end pb-1">
          <span className={`text-3xl font-bold num ${colorClass}`}>{pct}%</span>
        </div>
      </div>

      <p className="text-xs font-medium" style={{ color }}>{label}</p>
      <p className="text-xs text-gray-500 mt-1 text-center leading-relaxed">{ml.reason}</p>

      {/* Feature importance top */}
      {ml.top_feature && (
        <div className="mt-3 w-full bg-bg-700 rounded-lg p-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Top signal</span>
            <span className="text-gray-300 font-mono">{ml.top_feature}</span>
          </div>
          {ml.model_accuracy != null && (
            <div className="flex items-center justify-between mt-1">
              <span className="text-gray-500">Train accuracy</span>
              <span className="num text-gray-300">{ml.model_accuracy.toFixed(1)}%</span>
            </div>
          )}
        </div>
      )}

      {/* ML Prediction direction */}
      <div className={`mt-3 flex items-center gap-1.5 text-xs font-medium ${ml.trade ? 'text-gain' : 'text-loss'}`}>
        {ml.trade
          ? <><TrendingUp className="w-3.5 h-3.5" /> Favorable conditions today</>
          : <><TrendingDown className="w-3.5 h-3.5" /> Skip — unfavorable today</>
        }
      </div>
    </div>
  )
}
