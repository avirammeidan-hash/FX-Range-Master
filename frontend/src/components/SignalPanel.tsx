import { ArrowUpCircle, ArrowDownCircle, MinusCircle, AlertCircle } from 'lucide-react'
import { FxStatus } from '../services/api'

interface Props {
  data: FxStatus
}

interface Step {
  label: string
  price: number
  size: number
  note: string
}

function buildSteps(data: FxStatus): { action: 'BUY' | 'SELL' | null; steps: Step[]; tp: number; sl: number } {
  const { price, baseline, upper, lower, stop_upper, stop_lower, signal } = data
  const sig = signal?.signal

  if (sig === 'SELL' || (sig !== 'BUY' && price > upper)) {
    return {
      action: 'SELL',
      steps: [
        { label: 'Step 1 (40%)', price: upper,                                    size: 40, note: 'At upper band — best R/R' },
        { label: 'Step 2 (30%)', price: upper + (stop_upper - upper) * 0.5,       size: 30, note: 'Mid upper → stop' },
        { label: 'Step 3 (30%)', price: stop_upper - (stop_upper - upper) * 0.1,  size: 30, note: 'Near stop — best price' },
      ],
      tp: baseline,
      sl: stop_upper,
    }
  }

  if (sig === 'BUY' || (sig !== 'SELL' && price < lower)) {
    return {
      action: 'BUY',
      steps: [
        { label: 'Step 1 (40%)', price: lower,                                    size: 40, note: 'At lower band — best R/R' },
        { label: 'Step 2 (30%)', price: lower - (lower - stop_lower) * 0.5,       size: 30, note: 'Mid lower → stop' },
        { label: 'Step 3 (30%)', price: stop_lower + (lower - stop_lower) * 0.1,  size: 30, note: 'Near stop — best price' },
      ],
      tp: baseline,
      sl: stop_lower,
    }
  }

  return { action: null, steps: [], tp: baseline, sl: 0 }
}

export default function SignalPanel({ data }: Props) {
  const { action, steps, tp, sl } = buildSteps(data)
  const sig = data.signal?.signal ?? 'FLAT'

  // Distances for context
  const distUpper = data.dist_upper_pct
  const distLower = data.dist_lower_pct

  const headerColor =
    sig === 'BUY'  ? 'text-gain' :
    sig === 'SELL' ? 'text-loss' :
    sig === 'HOLD' ? 'text-warn' : 'text-gray-500'

  const HeaderIcon =
    sig === 'BUY'  ? ArrowUpCircle   :
    sig === 'SELL' ? ArrowDownCircle  :
    sig === 'HOLD' ? AlertCircle      : MinusCircle

  // Reasoning from tech data
  const tech = data.technical
  const reasons: string[] = []
  if (tech?.available) {
    if (tech.ema_cross === 'golden') reasons.push('Golden Cross ✓')
    if (tech.ema_cross === 'death')  reasons.push('Death Cross ✗')
    if (tech.rsi14 != null) {
      if (tech.rsi14 > 70) reasons.push(`RSI ${tech.rsi14.toFixed(0)} (overbought)`)
      else if (tech.rsi14 < 30) reasons.push(`RSI ${tech.rsi14.toFixed(0)} (oversold)`)
      else reasons.push(`RSI ${tech.rsi14.toFixed(0)}`)
    }
    if (tech.trend) reasons.push(`Trend: ${tech.trend}`)
  }
  if (data.news_sentiment?.sentiment !== 'NEUTRAL') {
    reasons.push(`News: ${data.news_sentiment.sentiment}`)
  }
  if (data.ml_prediction?.ml_available) {
    reasons.push(`ML: ${data.ml_prediction.trade ? 'Trade' : 'Skip'} (${Math.round((data.ml_prediction.confidence ?? 0) * 100)}%)`)
  }

  return (
    <div className="card space-y-4">
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <HeaderIcon className={`w-6 h-6 ${headerColor}`} />
        <div>
          <h3 className={`text-lg font-bold ${headerColor}`}>{sig}</h3>
          {data.signal?.reason && (
            <p className="text-xs text-gray-400">{data.signal.reason}</p>
          )}
        </div>
        <div className="ml-auto text-right text-xs text-gray-500">
          <div>↑ {distUpper.toFixed(3)}% to upper</div>
          <div>↓ {distLower.toFixed(3)}% to lower</div>
        </div>
      </div>

      {/* ── Reasoning ───────────────────────────────────────────────────── */}
      {reasons.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {reasons.map((r, i) => (
            <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-bg-700 text-gray-300">{r}</span>
          ))}
        </div>
      )}

      {/* ── Step quota ──────────────────────────────────────────────────── */}
      {action && steps.length > 0 ? (
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
            Strategy (example $100 account)
          </p>
          <div className="space-y-2">
            {steps.map((step, i) => (
              <div key={i} className="flex items-center gap-3 bg-bg-700 rounded-lg px-3 py-2">
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                  action === 'BUY' ? 'bg-gain/20 text-gain' : 'bg-loss/20 text-loss'
                }`}>{i + 1}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs text-gray-400">{step.label}</span>
                    <span className="num font-medium text-white">{step.price.toFixed(4)}</span>
                  </div>
                  <p className="text-xs text-gray-500 truncate">{step.note}</p>
                </div>
                <span className={`text-sm font-bold num ${
                  action === 'BUY' ? 'text-gain' : 'text-loss'
                }`}>${step.size}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 flex gap-4 text-xs">
            <div className="flex-1 bg-bg-700 rounded-lg px-3 py-2">
              <p className="text-gray-500">Take Profit</p>
              <p className="num font-bold text-gain">{tp.toFixed(4)}</p>
              <p className="text-gray-500">(return to baseline)</p>
            </div>
            <div className="flex-1 bg-bg-700 rounded-lg px-3 py-2">
              <p className="text-gray-500">Stop Loss</p>
              <p className="num font-bold text-loss">{sl.toFixed(4)}</p>
              <p className="text-gray-500">({data.params.stop_ext_pct}% beyond band)</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-bg-700 rounded-lg p-4 text-center text-sm text-gray-500">
          <p>Price is inside the range — watch for boundary approach</p>
          <div className="mt-2 flex justify-center gap-6 text-xs">
            <span>BUY zone ≤ <span className="num text-gain">{data.lower.toFixed(4)}</span></span>
            <span>SELL zone ≥ <span className="num text-loss">{data.upper.toFixed(4)}</span></span>
          </div>
        </div>
      )}

      {/* ── Today events ────────────────────────────────────────────────── */}
      {data.today_events?.length > 0 && (
        <div className="border-t border-bg-600 pt-3">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Today's Events</p>
          <div className="space-y-1">
            {data.today_events.slice(0, 3).map((ev, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                  ev.impact === 'HIGH' ? 'bg-loss' : ev.impact === 'MEDIUM' ? 'bg-warn' : 'bg-gray-600'
                }`} />
                <span className="text-gray-300 truncate">{ev.name}</span>
                <span className="text-gray-600 ml-auto">{ev.country}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
