import { useState } from 'react'
import { RefreshCw, RotateCcw } from 'lucide-react'
import { usePolling } from '../hooks/usePolling'
import { getStatus, retrain, resetBaseline } from '../services/api'

export default function Settings() {
  const { data: status } = usePolling('status', getStatus, 10000)
  const [retraining, setRetraining] = useState(false)
  const [retrainResult, setRetrainResult] = useState<string | null>(null)
  const [resetting, setResetting] = useState(false)

  const handleRetrain = async () => {
    setRetraining(true)
    setRetrainResult(null)
    try {
      const res = await retrain()
      setRetrainResult(res.ok ? '✓ Model retrained successfully' : `Error: ${res.message}`)
    } catch (e) {
      setRetrainResult(`Error: ${e}`)
    } finally {
      setRetraining(false)
    }
  }

  const handleReset = async () => {
    if (!confirm('Reset baseline to current market price?')) return
    setResetting(true)
    try {
      await resetBaseline()
    } finally {
      setResetting(false)
    }
  }

  return (
    <div className="space-y-4 max-w-lg">
      <h1 className="text-xl font-bold">Settings</h1>

      {/* Strategy params */}
      {status && (
        <div className="card space-y-3">
          <h3 className="text-sm font-medium text-gray-300">Strategy Parameters</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Pair</span>
              <span className="text-gray-200 font-medium">{status.pair}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Half-width</span>
              <span className="num text-gray-200">±{status.params.half_width_pct}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Stop extension</span>
              <span className="num text-gray-200">{status.params.stop_ext_pct}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Data source</span>
              <span className="text-gray-200 uppercase">{status.data_source}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">ML model age</span>
              <span className="num text-gray-200">
                {status.ml_prediction?.model_age_days != null
                  ? `${status.ml_prediction.model_age_days} day(s)`
                  : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">ML train accuracy</span>
              <span className="num text-gray-200">
                {status.ml_prediction?.model_accuracy != null
                  ? `${status.ml_prediction.model_accuracy.toFixed(1)}%`
                  : '—'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="card space-y-3">
        <h3 className="text-sm font-medium text-gray-300">Actions</h3>

        {/* Retrain */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-200">Retrain ML Model</p>
              <p className="text-xs text-gray-500">Force re-train the skip-day Random Forest classifier</p>
            </div>
            <button
              className="btn-primary flex items-center gap-1.5 text-sm"
              onClick={handleRetrain}
              disabled={retraining}
            >
              {retraining
                ? <><RefreshCw className="w-4 h-4 animate-spin" /> Training...</>
                : <><RefreshCw className="w-4 h-4" /> Retrain</>
              }
            </button>
          </div>
          {retrainResult && (
            <p className={`text-xs px-3 py-1.5 rounded-lg ${
              retrainResult.startsWith('✓') ? 'bg-gain/10 text-gain' : 'bg-loss/10 text-loss'
            }`}>{retrainResult}</p>
          )}
        </div>

        {/* Reset baseline */}
        <div className="pt-2 border-t border-bg-600">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-200">Reset Baseline</p>
              <p className="text-xs text-gray-500">Set today's baseline to the current market price</p>
            </div>
            <button
              className="btn-danger flex items-center gap-1.5 text-sm"
              onClick={handleReset}
              disabled={resetting}
            >
              <RotateCcw className={`w-4 h-4 ${resetting ? 'animate-spin' : ''}`} />
              Reset
            </button>
          </div>
        </div>
      </div>

      {/* About */}
      <div className="card space-y-2 text-xs text-gray-500">
        <p className="font-medium text-gray-400">FX Range Master · React UI</p>
        <p>ULTRON architecture × FX-Range-Master visual aesthetic</p>
        <p>React 18 + TypeScript + Tailwind + lightweight-charts</p>
        <p>Flask backend on port 5000 · Firestore · Cloud Run</p>
      </div>
    </div>
  )
}
