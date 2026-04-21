import { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, LineChart, Newspaper, BarChart3,
  History, Settings, Activity, Wifi, WifiOff,
} from 'lucide-react'
import { usePolling } from '../hooks/usePolling'
import { getStatus } from '../services/api'

const NAV = [
  { to: '/',           label: 'Dashboard',  icon: LayoutDashboard },
  { to: '/chart',      label: 'Chart',      icon: LineChart        },
  { to: '/news',       label: 'News',       icon: Newspaper        },
  { to: '/analysis',   label: 'Analysis',   icon: BarChart3        },
  { to: '/history',    label: 'History',    icon: History          },
  { to: '/settings',   label: 'Settings',   icon: Settings         },
]

export default function Layout({ children }: { children: ReactNode }) {
  const { data, isError } = usePolling('status', getStatus, 5000)

  const live = !isError && !!data

  return (
    <div className="flex h-screen overflow-hidden">
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside className="hidden md:flex w-56 bg-bg-800 border-r border-bg-600 flex-col">
        {/* Logo */}
        <div className="p-5 border-b border-bg-600">
          <h1 className="text-lg font-bold flex items-center gap-2 text-accent">
            <Activity className="w-5 h-5" />
            FX Range Master
          </h1>
          <p className="text-xs text-gray-500 mt-0.5">USD / ILS · Intraday</p>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-3">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-5 py-2.5 text-sm transition-colors ${
                  isActive
                    ? 'bg-accent/10 text-accent border-r-2 border-accent'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-bg-700/50'
                }`
              }
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Status footer */}
        <div className="p-4 border-t border-bg-600 space-y-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-gray-500">Feed</span>
            <span className="flex items-center gap-1">
              {live
                ? <><Wifi className="w-3 h-3 text-gain" /><span className="text-gain">Live</span></>
                : <><WifiOff className="w-3 h-3 text-gray-600" /><span className="text-gray-600">Offline</span></>
              }
            </span>
          </div>
          {data && (
            <>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Source</span>
                <span className="text-gray-300 uppercase">{data.data_source}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">ML</span>
                <span className={data.ml_prediction?.ml_available ? 'text-gain' : 'text-gray-500'}>
                  {data.ml_prediction?.ml_available ? 'Active' : 'Unavail.'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Rec</span>
                <span className={
                  data.trade_recommendation === 'TRADE' ? 'text-gain' :
                  data.trade_recommendation === 'SKIP'  ? 'text-loss' : 'text-warn'
                }>
                  {data.trade_recommendation}
                </span>
              </div>
            </>
          )}
        </div>
      </aside>

      {/* ── Main ────────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto p-4 md:p-6">
        {children}
      </main>
    </div>
  )
}
