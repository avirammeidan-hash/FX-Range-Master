import { useEffect, useRef } from 'react'
import {
  createChart, ColorType, IChartApi,
  CrosshairMode,
} from 'lightweight-charts'
import { Candle } from '../services/api'

interface Props {
  candles: Candle[]
  upper?: number
  lower?: number
  stop_upper?: number
  stop_lower?: number
  baseline?: number
  ema20?: number[]
  ema50?: number[]
  height?: number
}

export default function CandleChart({
  candles,
  upper,
  lower,
  stop_upper,
  stop_lower,
  baseline,
  height = 380,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return

    // Cleanup previous
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#64748b',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#4f8eff', labelBackgroundColor: '#1e3a8a' },
        horzLine: { color: '#4f8eff', labelBackgroundColor: '#1e3a8a' },
      },
      timeScale: {
        borderColor: '#1e293b',
        timeVisible: true,
        secondsVisible: false,
      },
      rightPriceScale: {
        borderColor: '#1e293b',
        textColor: '#64748b',
      },
    })

    chartRef.current = chart

    // ── Candlestick series ────────────────────────────────────────────
    const candleSeries = chart.addCandlestickSeries({
      upColor:   '#22c55e',
      downColor: '#ef4444',
      borderUpColor:   '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor:   '#22c55e',
      wickDownColor: '#ef4444',
    })

    const candleData = candles.map(c => ({
      time: c.t.slice(0, 10) as never,
      open:  c.o,
      high:  c.h,
      low:   c.l,
      close: c.c,
    }))
    candleSeries.setData(candleData)

    // ── Horizontal price lines ────────────────────────────────────────
    if (upper != null) {
      candleSeries.createPriceLine({ price: upper,      color: '#ef4444', lineWidth: 1, lineStyle: 2, title: 'Upper' })
    }
    if (lower != null) {
      candleSeries.createPriceLine({ price: lower,      color: '#22c55e', lineWidth: 1, lineStyle: 2, title: 'Lower' })
    }
    if (baseline != null) {
      candleSeries.createPriceLine({ price: baseline,   color: '#4f8eff', lineWidth: 1, lineStyle: 1, title: 'Base' })
    }
    if (stop_upper != null) {
      candleSeries.createPriceLine({ price: stop_upper, color: '#7f1d1d', lineWidth: 1, lineStyle: 3, title: 'SL↑' })
    }
    if (stop_lower != null) {
      candleSeries.createPriceLine({ price: stop_lower, color: '#14532d', lineWidth: 1, lineStyle: 3, title: 'SL↓' })
    }

    chart.timeScale().fitContent()

    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [candles, upper, lower, stop_upper, stop_lower, baseline, height])

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(() => {
      if (chartRef.current && containerRef.current) {
        chartRef.current.resize(containerRef.current.clientWidth, height)
      }
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [height])

  return (
    <div className="w-full">
      {candles.length === 0 ? (
        <div
          className="flex items-center justify-center text-gray-600 text-sm bg-bg-700 rounded-lg"
          style={{ height }}
        >
          No candle data
        </div>
      ) : (
        <div ref={containerRef} className="w-full" style={{ height }} />
      )}
    </div>
  )
}
