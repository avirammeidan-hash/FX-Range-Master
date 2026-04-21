import { RefreshCw, AlertTriangle, CheckCircle, Info } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { usePolling } from '../hooks/usePolling'
import { getSignalPerformance } from '../services/api'
import { AccuracyGauge } from '@trading/ui'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, CartesianGrid, ReferenceLine, Legend,
} from 'recharts'

// ── Tooltip styles ────────────────────────────────────────────────────────────
const TT_STYLE = { background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8 }
const TT_LABEL = { color: '#e2e8f0' }

export default function Performance() {
  const qc = useQueryClient()
  const { data, isLoading, isError } = usePolling('signal-perf', getSignalPerformance, 60_000)

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <RefreshCw className="w-6 h-6 animate-spin text-accent mr-2" />
      <span className="text-gray-400">Loading performance data…</span>
    </div>
  )

  if (isError || data?.error) return (
    <div className="card text-center py-12 space-y-2">
      <AlertTriangle className="w-8 h-8 text-loss mx-auto" />
      <p className="text-loss font-medium">No data yet</p>
      <p className="text-xs text-gray-500">{data?.error ?? 'Keep the app running — predictions are scored every 10/30/60 min'}</p>
    </div>
  )

  if (!data) return null

  const { calibration, accuracy_by_day, pnl_curve, pnl_summary, signal_breakdown, hints } = data
  const pnlColor   = (pnl_summary.return_pct ?? 0) >= 0 ? '#22c55e' : '#ef4444'
  const tradeBreak = signal_breakdown['TRADE']
  const skipBreak  = signal_breakdown['SKIP']

  return (
    <div className="space-y-5">

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">AI Performance Research</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            {data.total_records} scored predictions · 30-minute window
          </p>
        </div>
        <button className="btn-ghost flex items-center gap-1.5 text-xs"
          onClick={() => qc.invalidateQueries({ queryKey: ['signal-perf'] })}>
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {/* ── Summary cards ───────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {/* TRADE accuracy */}
        <div className="card text-center space-y-2">
          <p className="text-xs text-gray-500">TRADE Accuracy</p>
          {tradeBreak ? (
            <>
              <AccuracyGauge pct={tradeBreak.accuracy} size="sm" threshold={60} />
              <p className="text-xs text-gray-500">{tradeBreak.wins}W / {tradeBreak.losses}L</p>
            </>
          ) : <p className="text-gray-600 text-sm py-4">—</p>}
        </div>

        {/* SKIP accuracy */}
        <div className="card text-center space-y-2">
          <p className="text-xs text-gray-500">SKIP Accuracy</p>
          {skipBreak ? (
            <>
              <AccuracyGauge pct={skipBreak.accuracy} size="sm" threshold={60} />
              <p className="text-xs text-gray-500">{skipBreak.wins}W / {skipBreak.losses}L</p>
            </>
          ) : <p className="text-gray-600 text-sm py-4">—</p>}
        </div>

        {/* P&L */}
        <div className="card text-center space-y-1 flex flex-col justify-center">
          <p className="text-xs text-gray-500">P&L Simulation</p>
          <p className="text-xs text-gray-600">$10/trade on $100</p>
          <p className={`text-3xl font-bold num ${pnl_summary.return_pct >= 0 ? 'text-gain' : 'text-loss'}`}>
            {pnl_summary.return_pct >= 0 ? '+' : ''}${pnl_summary.return_pct.toFixed(2)}
          </p>
          <p className="text-xs text-gray-500">
            {pnl_summary.win_rate.toFixed(0)}% win rate · {pnl_summary.total_trades} trades
          </p>
        </div>

        {/* Hints count */}
        <div className="card text-center space-y-1 flex flex-col justify-center">
          <p className="text-xs text-gray-500">Issues Found</p>
          <p className="text-3xl font-bold" style={{
            color: hints.filter(h => h.severity === 'high').length > 0 ? '#ef4444' :
                   hints.length > 0 ? '#f59e0b' : '#22c55e'
          }}>
            {hints.length}
          </p>
          <p className="text-xs text-gray-500">
            {hints.filter(h => h.severity === 'high').length} high priority
          </p>
        </div>
      </div>

      {/* ── Confidence Calibration ───────────────────────────────────────────── */}
      {calibration.length > 0 && (
        <div className="card space-y-3">
          <div>
            <h3 className="text-sm font-medium text-gray-300">Confidence Calibration</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Does 75% confidence actually mean 75% correct? Bars above the line = better than claimed.
            </p>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={calibration} margin={{ left: 0, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="bucket" tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} unit="%" />
              <Tooltip
                contentStyle={TT_STYLE} labelStyle={TT_LABEL}
                formatter={(v: number, name: string) => [`${v.toFixed(1)}%`, name === 'actual_pct' ? 'Actual accuracy' : 'Model confidence']}
              />
              <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }}
                formatter={(val) => val === 'actual_pct' ? 'Actual accuracy' : 'Model confidence'} />
              {/* Ideal diagonal reference — for a perfectly calibrated model all bars equal */}
              <Bar dataKey="expected_pct" name="expected_pct" fill="#1e3a5f" radius={[4, 4, 0, 0]} />
              <Bar dataKey="actual_pct"   name="actual_pct"   radius={[4, 4, 0, 0]}>
                {calibration.map((c, i) => (
                  <Cell key={i} fill={c.gap > 0 ? '#22c55e' : c.gap < -10 ? '#ef4444' : '#f59e0b'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {/* Calibration table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-bg-600">
                  <th className="text-left py-1">Confidence bucket</th>
                  <th className="text-right py-1">Model says</th>
                  <th className="text-right py-1">Actually correct</th>
                  <th className="text-right py-1">Gap</th>
                  <th className="text-right py-1">Samples</th>
                </tr>
              </thead>
              <tbody>
                {calibration.map((c, i) => (
                  <tr key={i} className="border-b border-bg-700">
                    <td className="py-1.5 text-gray-300">{c.bucket}</td>
                    <td className="py-1.5 text-right num text-gray-400">{c.expected_pct.toFixed(1)}%</td>
                    <td className={`py-1.5 text-right num font-medium ${
                      c.actual_pct >= 60 ? 'text-gain' : c.actual_pct >= 45 ? 'text-warn' : 'text-loss'
                    }`}>{c.actual_pct.toFixed(1)}%</td>
                    <td className={`py-1.5 text-right num ${c.gap > 0 ? 'text-gain' : 'text-loss'}`}>
                      {c.gap > 0 ? '+' : ''}{c.gap.toFixed(1)}%
                    </td>
                    <td className="py-1.5 text-right text-gray-500">{c.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── P&L Simulation ───────────────────────────────────────────────────── */}
      {pnl_curve.length > 1 && (
        <div className="card space-y-3">
          <div>
            <h3 className="text-sm font-medium text-gray-300">P&L Simulation</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              $100 starting capital · $10 per TRADE signal · gain/loss = actual price move × position
            </p>
          </div>
          <div className="flex items-center gap-6 text-sm">
            <div>
              <span className="text-gray-500 text-xs">Start</span>
              <p className="num font-bold text-gray-300">${pnl_summary.start_equity}</p>
            </div>
            <div>
              <span className="text-gray-500 text-xs">End</span>
              <p className="num font-bold" style={{ color: pnlColor }}>${pnl_summary.end_equity.toFixed(2)}</p>
            </div>
            <div>
              <span className="text-gray-500 text-xs">Return</span>
              <p className={`num font-bold ${pnl_summary.return_pct >= 0 ? 'text-gain' : 'text-loss'}`}>
                {pnl_summary.return_pct >= 0 ? '+' : ''}${pnl_summary.return_pct.toFixed(2)}
              </p>
            </div>
            <div>
              <span className="text-gray-500 text-xs">Win rate</span>
              <p className="num font-bold text-gray-300">{pnl_summary.win_rate.toFixed(0)}%</p>
            </div>
            <div>
              <span className="text-gray-500 text-xs">Trades</span>
              <p className="num font-bold text-gray-300">{pnl_summary.total_trades}</p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={pnl_curve} margin={{ left: 0, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 9 }}
                interval="preserveStartEnd" />
              <YAxis domain={['auto', 'auto']} tick={{ fill: '#64748b', fontSize: 10 }}
                tickFormatter={(v) => `$${v}`} />
              <Tooltip contentStyle={TT_STYLE} labelStyle={TT_LABEL}
                formatter={(v: number) => [`$${v.toFixed(2)}`, 'Equity']} />
              <ReferenceLine y={100} stroke="#4f8eff" strokeDasharray="4 4" label={{ value: '$100', fill: '#4f8eff', fontSize: 10 }} />
              <Line type="monotone" dataKey="equity" stroke={pnlColor}
                strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Accuracy over time ───────────────────────────────────────────────── */}
      {accuracy_by_day.length > 1 && (
        <div className="card space-y-3">
          <h3 className="text-sm font-medium text-gray-300">Accuracy Over Time</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={accuracy_by_day} margin={{ left: 0, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 9 }} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} unit="%" />
              <Tooltip contentStyle={TT_STYLE} labelStyle={TT_LABEL}
                formatter={(v: number, _, p) => [`${v.toFixed(1)}% (${p.payload.total} samples)`, 'Accuracy']} />
              <ReferenceLine y={50} stroke="#475569" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="accuracy" stroke="#4f8eff"
                strokeWidth={2} dot={{ fill: '#4f8eff', r: 3 }} activeDot={{ r: 5 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Improvement hints ────────────────────────────────────────────────── */}
      {hints.length > 0 && (
        <div className="card space-y-3">
          <h3 className="text-sm font-medium text-gray-300">What to Improve</h3>
          <div className="space-y-2">
            {hints
              .sort((a, b) =>
                (a.severity === 'high' ? 0 : a.severity === 'medium' ? 1 : 2) -
                (b.severity === 'high' ? 0 : b.severity === 'medium' ? 1 : 2))
              .map((h, i) => {
                const Icon  = h.severity === 'high' ? AlertTriangle : h.severity === 'low' ? CheckCircle : Info
                const color = h.severity === 'high' ? 'text-loss border-loss/30 bg-loss/5' :
                              h.severity === 'low'  ? 'text-gain border-gain/30 bg-gain/5' :
                                                      'text-warn border-warn/30 bg-warn/5'
                return (
                  <div key={i} className={`flex items-start gap-2.5 border rounded-lg px-3 py-2.5 ${color}`}>
                    <Icon className="w-4 h-4 flex-shrink-0 mt-0.5" />
                    <p className="text-xs leading-relaxed">{h.message}</p>
                  </div>
                )
              })}
          </div>
        </div>
      )}

      {hints.length === 0 && data.total_records >= 50 && (
        <div className="card flex items-center gap-3 text-gain">
          <CheckCircle className="w-5 h-5 flex-shrink-0" />
          <p className="text-sm">No issues detected — model is well calibrated and profitable. Keep monitoring.</p>
        </div>
      )}
    </div>
  )
}
