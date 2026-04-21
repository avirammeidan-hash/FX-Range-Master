/**
 * FX-specific CandleChart wrapper.
 * Delegates to @trading/ui/CandleChart and pre-wires FX price lines
 * (upper/lower bands, stop levels, baseline).
 */
import { CandleChart as SharedChart } from '@trading/ui'
import type { PriceLine, Candle } from '@trading/api-client'

interface Props {
  candles: Candle[]
  upper?: number
  lower?: number
  stop_upper?: number
  stop_lower?: number
  baseline?: number
  height?: number
}

export default function CandleChart({
  candles, upper, lower, stop_upper, stop_lower, baseline, height = 380,
}: Props) {
  const priceLines: PriceLine[] = [
    ...(upper      != null ? [{ price: upper,      color: '#ef4444', title: 'Upper', lineStyle: 2 as const }] : []),
    ...(lower      != null ? [{ price: lower,      color: '#22c55e', title: 'Lower', lineStyle: 2 as const }] : []),
    ...(baseline   != null ? [{ price: baseline,   color: '#4f8eff', title: 'Base',  lineStyle: 1 as const }] : []),
    ...(stop_upper != null ? [{ price: stop_upper, color: '#7f1d1d', title: 'SL↑',  lineStyle: 3 as const }] : []),
    ...(stop_lower != null ? [{ price: stop_lower, color: '#14532d', title: 'SL↓',  lineStyle: 3 as const }] : []),
  ]
  return <SharedChart candles={candles} priceLines={priceLines} height={height} />
}
