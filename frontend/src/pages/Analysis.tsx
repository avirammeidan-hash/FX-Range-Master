import { RefreshCw, Brain, Calendar } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { usePolling } from '../hooks/usePolling'
import { getAiPerformance, getMlExport, getCalendar } from '../services/api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export default function Analysis() {
  const qc = useQueryClient()
  const { data: perf }     = usePolling('ai-perf',  getAiPerformance, 30000)
  const { data: ml }       = usePolling('ml-export', getMlExport,     60000)
  const { data: calendar } = usePolling('calendar',  getCalendar,      60000)

  const featureData = ml?.feature_importance
    ? Object.entries(ml.feature_importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([name, val]) => ({ name, value: +(val * 100).toFixed(1) }))
    : []

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">Analysis</h1>
        <button className="btn-ghost flex items-center gap-1.5 text-xs"
          onClick={() => qc.invalidateQueries()}>
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {/* ── AI Performance ───────────────────────────────────────────────── */}
      <div className="card space-y-3">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-accent" />
          <h3 className="text-sm font-medium text-gray-300">AI Decision Performance</h3>
        </div>

        {!perf?.summary ? (
          <p className="text-sm text-gray-500">No performance data yet</p>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { label: 'Overall',       val: perf.summary.accuracy_pct,      color: perf.summary.accuracy_pct >= 60 ? 'text-gain' : 'text-warn' },
              { label: 'TRADE signals', val: perf.summary.trade_accuracy_pct, color: 'text-accent' },
              { label: 'SKIP signals',  val: perf.summary.skip_accuracy_pct,  color: 'text-cyan'   },
              { label: 'Sample size',   val: perf.summary.total,              color: 'text-gray-300', suffix: '' },
            ].map(({ label, val, color, suffix }) => (
              <div key={label} className="bg-bg-700 rounded-lg p-3 text-center">
                <p className="text-xs text-gray-500 mb-1">{label}</p>
                <p className={`text-2xl font-bold num ${color}`}>
                  {typeof val === 'number' ? val.toFixed(suffix !== '' ? 1 : 0) : val}
                  {suffix ?? '%'}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* Recent decisions */}
        {perf?.recent && perf.recent.length > 0 && (
          <div className="mt-2">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Recent Decisions</p>
            <div className="space-y-1.5 max-h-48 overflow-y-auto no-scrollbar">
              {perf.recent.slice(0, 15).map((d, i) => (
                <div key={i} className="flex items-center gap-3 bg-bg-700 rounded-lg px-3 py-1.5 text-xs">
                  <span className={`font-bold ${d.ml_decision === 'TRADE' ? 'text-gain' : 'text-loss'}`}>
                    {d.ml_decision}
                  </span>
                  <span className="num text-gray-400">{(d.confidence * 100).toFixed(0)}%</span>
                  <span className="text-gray-600">{d.timestamp?.slice(0, 16).replace('T', ' ')}</span>
                  <span className={`ml-auto font-medium ${
                    d.correct === true  ? 'text-gain' :
                    d.correct === false ? 'text-loss' : 'text-gray-500'
                  }`}>
                    {d.correct === true ? '✓' : d.correct === false ? '✗' : '—'}
                  </span>
                  <span className={`num ${(d.change_pct ?? 0) >= 0 ? 'text-gain' : 'text-loss'}`}>
                    {(d.change_pct ?? 0) >= 0 ? '+' : ''}{(d.change_pct ?? 0).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── ML Feature Importance ────────────────────────────────────────── */}
      {featureData.length > 0 && (
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-medium text-gray-300">ML Feature Importance</h3>
            {ml?.accuracy != null && (
              <span className="badge-blue">Train acc: {ml.accuracy.toFixed(1)}%</span>
            )}
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={featureData} layout="vertical" margin={{ left: 8, right: 8 }}>
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} unit="%" />
              <YAxis type="category" dataKey="name" width={110} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }}
                labelStyle={{ color: '#e2e8f0' }}
                formatter={(v: number) => [`${v}%`, 'Importance']}
              />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {featureData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#4f8eff' : i < 3 ? '#06b6d4' : '#1e40af'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {ml?.train_date && (
            <p className="text-xs text-gray-500">
              Trained: {new Date(ml.train_date).toLocaleDateString()} ·
              Window: {ml.training_days ?? '—'} days ·
              Threshold: {((ml.threshold ?? 0) * 100).toFixed(0)}%
            </p>
          )}
        </div>
      )}

      {/* ── Economic Calendar ─────────────────────────────────────────────── */}
      <div className="card space-y-3">
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-accent" />
          <h3 className="text-sm font-medium text-gray-300">Economic Calendar</h3>
        </div>
        {!calendar?.events?.length ? (
          <p className="text-sm text-gray-500">No upcoming events</p>
        ) : (
          <div className="space-y-2">
            {calendar.events.slice(0, 10).map((ev, i) => (
              <div key={i} className="flex items-start gap-3 bg-bg-700 rounded-lg px-3 py-2">
                <span className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${
                  ev.impact === 'HIGH'   ? 'bg-loss' :
                  ev.impact === 'MEDIUM' ? 'bg-warn' : 'bg-gray-600'
                }`} />
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-gray-200 font-medium">{ev.name}</p>
                  <div className="flex items-center gap-3 mt-0.5 text-xs text-gray-500">
                    <span>{ev.date?.slice(0, 10)}</span>
                    {ev.country && <span>{ev.country}</span>}
                    {ev.forecast && <span>Forecast: <span className="text-gray-300">{ev.forecast}</span></span>}
                    {ev.actual   && <span>Actual: <span className={`font-medium ${
                      ev.actual > ev.forecast! ? 'text-gain' : 'text-loss'
                    }`}>{ev.actual}</span></span>}
                  </div>
                </div>
                <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
                  ev.impact === 'HIGH'   ? 'bg-loss/20 text-loss' :
                  ev.impact === 'MEDIUM' ? 'bg-warn/20 text-warn' :
                  'bg-bg-600 text-gray-500'
                }`}>{ev.impact}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
