import { useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { usePolling } from '../hooks/usePolling'
import { getCandles, getStatus } from '../services/api'
import CandleChart from '../components/CandleChart'

export default function ChartPage() {
  const qc = useQueryClient()
  const [showLevels, setShowLevels] = useState(true)

  const { data: candles, isLoading: loadingCandles } = usePolling('candles', getCandles, 60000)
  const { data: status } = usePolling('status', getStatus, 5000)

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-bold">Chart</h1>
          <p className="text-xs text-gray-500">USD/ILS · Intraday candles</p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={showLevels}
              onChange={e => setShowLevels(e.target.checked)}
              className="accent-accent-500"
            />
            Range levels
          </label>
          <button
            className="btn-ghost flex items-center gap-1.5 text-xs"
            onClick={() => {
              qc.invalidateQueries({ queryKey: ['candles'] })
              qc.invalidateQueries({ queryKey: ['status'] })
            }}
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh
          </button>
        </div>
      </div>

      {/* Chart */}
      <div className="card p-4">
        {loadingCandles ? (
          <div className="flex items-center justify-center h-96 text-gray-500 text-sm">
            <RefreshCw className="w-5 h-5 animate-spin mr-2" /> Loading candles...
          </div>
        ) : (
          <CandleChart
            candles={candles?.candles ?? []}
            upper={showLevels ? status?.upper : undefined}
            lower={showLevels ? status?.lower : undefined}
            stop_upper={showLevels ? status?.stop_upper : undefined}
            stop_lower={showLevels ? status?.stop_lower : undefined}
            baseline={showLevels ? status?.baseline : undefined}
            height={450}
          />
        )}
      </div>

      {/* Level legend */}
      {showLevels && status && (
        <div className="card-sm">
          <div className="flex flex-wrap gap-4 text-xs num">
            {[
              { label: 'Stop↑', val: status.stop_upper, color: 'text-red-700' },
              { label: 'Upper',  val: status.upper,      color: 'text-loss'    },
              { label: 'Base',   val: status.baseline,   color: 'text-accent'  },
              { label: 'Lower',  val: status.lower,      color: 'text-gain'    },
              { label: 'Stop↓', val: status.stop_lower, color: 'text-green-700'},
            ].map(({ label, val, color }) => (
              <div key={label} className="flex items-center gap-1.5">
                <span className="text-gray-500">{label}</span>
                <span className={`font-medium ${color}`}>{val.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
