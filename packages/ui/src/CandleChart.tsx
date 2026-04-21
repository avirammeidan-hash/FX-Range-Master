import { useEffect, useRef } from 'react'
import { createChart, ColorType, IChartApi, CrosshairMode } from 'lightweight-charts'
import type { Candle, PriceLine } from '@trading/api-client'

interface Props {
  candles: Candle[]
  /** Optional horizontal price lines (support/resistance, bands, etc.) */
  priceLines?: PriceLine[]
  height?: number
  /** Crosshair / accent color — defaults to #4f8eff */
  accentColor?: string
}

/**
 * Generic candlestick chart using lightweight-charts v4.
 * Works for any asset — FX pairs, stocks, crypto.
 *
 * @example
 * <CandleChart
 *   candles={candles}
 *   priceLines={[
 *     { price: 3.168, color: '#22c55e', title: 'Support' },
 *     { price: 3.195, color: '#ef4444', title: 'Resistance' },
 *   ]}
 * />
 */
export default function CandleChart({
  candles,
  priceLines = [],
  height = 380,
  accentColor = '#4f8eff',
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef     = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return

    if (chartRef.current) { chartRef.current.remove(); chartRef.current = null }

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
        vertLine: { color: accentColor, labelBackgroundColor: '#1e3a8a' },
        horzLine: { color: accentColor, labelBackgroundColor: '#1e3a8a' },
      },
      timeScale: { borderColor: '#1e293b', timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: '#1e293b', textColor: '#64748b' },
    })

    chartRef.current = chart

    const series = chart.addCandlestickSeries({
      upColor:        '#22c55e',
      downColor:      '#ef4444',
      borderUpColor:  '#22c55e',
      borderDownColor:'#ef4444',
      wickUpColor:    '#22c55e',
      wickDownColor:  '#ef4444',
    })

    series.setData(
      candles.map(c => ({
        time:  c.t.slice(0, 10) as never,
        open:  c.o,
        high:  c.h,
        low:   c.l,
        close: c.c,
      }))
    )

    priceLines.forEach(pl => {
      series.createPriceLine({
        price:      pl.price,
        color:      pl.color,
        lineWidth:  pl.lineWidth ?? 1,
        lineStyle:  pl.lineStyle ?? 2,
        title:      pl.title,
      })
    })

    chart.timeScale().fitContent()

    return () => { if (chartRef.current) { chartRef.current.remove(); chartRef.current = null } }
  }, [candles, priceLines, height, accentColor])

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(() => {
      if (chartRef.current && containerRef.current)
        chartRef.current.resize(containerRef.current.clientWidth, height)
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [height])

  if (candles.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-gray-600 text-sm bg-bg-700 rounded-lg"
        style={{ height }}
      >
        No candle data
      </div>
    )
  }

  return <div ref={containerRef} className="w-full" style={{ height }} />
}
