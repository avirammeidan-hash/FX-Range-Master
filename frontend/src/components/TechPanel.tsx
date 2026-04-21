import { TechnicalData } from '../services/api'

interface Props {
  tech: TechnicalData
}

export default function TechPanel({ tech }: Props) {
  if (!tech.available) {
    return (
      <div className="card">
        <h3 className="text-sm font-medium text-gray-400 mb-2">Technical Analysis</h3>
        <p className="text-sm text-gray-600">TA data unavailable</p>
      </div>
    )
  }

  const rsi = tech.rsi14 ?? 50
  const rsiColor =
    rsi > 70 ? 'text-loss' :
    rsi < 30 ? 'text-gain' : 'text-gray-300'

  const crossLabel = tech.ema_cross === 'golden'
    ? { text: 'Golden Cross', cls: 'badge-green' }
    : tech.ema_cross === 'death'
    ? { text: 'Death Cross', cls: 'badge-red' }
    : { text: 'Neutral', cls: 'badge-blue' }

  return (
    <div className="card space-y-3">
      <h3 className="text-sm font-medium text-gray-400">Technical Analysis</h3>

      {/* RSI */}
      <div>
        <div className="flex items-center justify-between text-xs mb-1">
          <span className="text-gray-500">RSI (14)</span>
          <span className={`num font-bold ${rsiColor}`}>{rsi.toFixed(1)}</span>
        </div>
        <div className="relative h-2 bg-bg-700 rounded-full overflow-hidden">
          {/* Overbought/oversold zones */}
          <div className="absolute left-0 top-0 h-full w-[30%] bg-gain/10 rounded-l-full" />
          <div className="absolute right-0 top-0 h-full w-[30%] bg-loss/10 rounded-r-full" />
          {/* RSI bar */}
          <div
            className="absolute left-0 top-0 h-full rounded-full transition-all duration-500"
            style={{
              width: `${rsi}%`,
              background: rsi > 70 ? '#ef4444' : rsi < 30 ? '#22c55e' : '#4f8eff',
            }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-600 mt-0.5">
          <span>0</span>
          <span className="text-gray-500">30</span>
          <span className="text-gray-500">70</span>
          <span>100</span>
        </div>
      </div>

      {/* EMA cross */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-500">EMA 20 / 50</p>
          <div className="flex items-baseline gap-2 num text-xs mt-0.5">
            <span className="text-cyan">{tech.ema20?.toFixed(4) ?? '—'}</span>
            <span className="text-gray-600">/</span>
            <span className="text-warn">{tech.ema50?.toFixed(4) ?? '—'}</span>
          </div>
        </div>
        <span className={crossLabel.cls}>{crossLabel.text}</span>
      </div>

      {/* Bollinger Bands */}
      {tech.bb_upper != null && (
        <div>
          <p className="text-xs text-gray-500 mb-1">Bollinger Bands</p>
          <div className="space-y-0.5 num text-xs">
            <div className="flex justify-between">
              <span className="text-gray-500">Upper</span>
              <span className="text-loss">{tech.bb_upper.toFixed(4)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Mid</span>
              <span className="text-gray-300">{tech.bb_mid?.toFixed(4) ?? '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Lower</span>
              <span className="text-gain">{tech.bb_lower?.toFixed(4) ?? '—'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Trend + ATR */}
      <div className="flex gap-3">
        {tech.trend && (
          <div className="flex-1 bg-bg-700 rounded-lg p-2 text-center">
            <p className="text-xs text-gray-500">Trend</p>
            <p className={`text-sm font-bold ${
              tech.trend === 'UP' ? 'text-gain' :
              tech.trend === 'DOWN' ? 'text-loss' : 'text-gray-400'
            }`}>{tech.trend}</p>
          </div>
        )}
        {tech.atr != null && (
          <div className="flex-1 bg-bg-700 rounded-lg p-2 text-center">
            <p className="text-xs text-gray-500">ATR</p>
            <p className="text-sm font-bold num text-gray-300">{tech.atr.toFixed(4)}</p>
          </div>
        )}
        {tech.momentum != null && (
          <div className="flex-1 bg-bg-700 rounded-lg p-2 text-center">
            <p className="text-xs text-gray-500">Mom 5d</p>
            <p className={`text-sm font-bold num ${tech.momentum > 0 ? 'text-gain' : 'text-loss'}`}>
              {tech.momentum >= 0 ? '+' : ''}{tech.momentum.toFixed(2)}%
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
