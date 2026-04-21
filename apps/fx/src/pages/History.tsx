import { usePolling } from '../hooks/usePolling'
import { getAiPerformance } from '../services/api'
import { formatDistanceToNow } from 'date-fns'
import { CheckCircle, XCircle, MinusCircle, TrendingUp, TrendingDown } from 'lucide-react'

export default function History() {
  const { data: perf, isLoading } = usePolling('ai-perf', getAiPerformance, 30000)

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Decision History</h1>

      {/* Summary row */}
      {perf?.summary && (
        <div className="grid grid-cols-3 gap-3">
          <div className="card-sm text-center">
            <p className="text-xs text-gray-500 mb-1">Overall accuracy</p>
            <p className={`text-2xl font-bold num ${perf.summary.accuracy_pct >= 60 ? 'text-gain' : 'text-warn'}`}>
              {perf.summary.accuracy_pct.toFixed(1)}%
            </p>
          </div>
          <div className="card-sm text-center">
            <p className="text-xs text-gray-500 mb-1">Total decisions</p>
            <p className="text-2xl font-bold num text-gray-200">{perf.summary.total}</p>
          </div>
          <div className="card-sm text-center">
            <p className="text-xs text-gray-500 mb-1">Period</p>
            <p className="text-2xl font-bold num text-gray-200">{perf.summary.period_days}d</p>
          </div>
        </div>
      )}

      {/* Decision table */}
      <div className="card p-0 overflow-hidden">
        <div className="px-4 py-3 border-b border-bg-600">
          <h3 className="text-sm font-medium text-gray-300">All ML Decisions</h3>
        </div>
        {isLoading ? (
          <div className="p-8 text-center text-gray-500 text-sm">Loading...</div>
        ) : !perf?.recent?.length ? (
          <div className="p-8 text-center text-gray-500 text-sm">No recorded decisions yet</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-bg-600 text-gray-500">
                  <th className="text-left px-4 py-2 font-medium">Time</th>
                  <th className="text-left px-4 py-2 font-medium">Decision</th>
                  <th className="text-right px-4 py-2 font-medium">Confidence</th>
                  <th className="text-right px-4 py-2 font-medium">Price then</th>
                  <th className="text-right px-4 py-2 font-medium">30m move</th>
                  <th className="text-center px-4 py-2 font-medium">Correct?</th>
                </tr>
              </thead>
              <tbody>
                {perf.recent.map((d, i) => {
                  const chg = d.change_pct ?? 0
                  const ChgIcon = chg > 0 ? TrendingUp : chg < 0 ? TrendingDown : MinusCircle
                  const ResultIcon = d.correct === true
                    ? CheckCircle : d.correct === false
                    ? XCircle : MinusCircle

                  return (
                    <tr key={i} className="border-b border-bg-700/50 hover:bg-bg-700/30 transition-colors">
                      <td className="px-4 py-2 text-gray-500">
                        {d.timestamp
                          ? formatDistanceToNow(new Date(d.timestamp), { addSuffix: true })
                          : '—'}
                      </td>
                      <td className="px-4 py-2">
                        <span className={`font-bold ${d.ml_decision === 'TRADE' ? 'text-gain' : 'text-loss'}`}>
                          {d.ml_decision}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right num text-gray-300">
                        {((d.confidence ?? 0) * 100).toFixed(0)}%
                      </td>
                      <td className="px-4 py-2 text-right num text-gray-300">
                        {d.price_at?.toFixed(4) ?? '—'}
                      </td>
                      <td className="px-4 py-2 text-right">
                        <span className={`flex items-center justify-end gap-0.5 num ${chg >= 0 ? 'text-gain' : 'text-loss'}`}>
                          <ChgIcon className="w-3 h-3" />
                          {chg >= 0 ? '+' : ''}{chg.toFixed(3)}%
                        </span>
                      </td>
                      <td className="px-4 py-2 text-center">
                        <ResultIcon className={`w-4 h-4 mx-auto ${
                          d.correct === true  ? 'text-gain' :
                          d.correct === false ? 'text-loss' : 'text-gray-600'
                        }`} />
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
