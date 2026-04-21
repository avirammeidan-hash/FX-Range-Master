import { NavLink } from 'react-router-dom'
import { BarChart2, Star, Newspaper, Settings, Activity } from 'lucide-react'
import { useWebSocket } from '@trading/hooks'

const NAV = [
  { to: '/stocks',    label: 'Dashboard', icon: BarChart2 },
  { to: '/watchlist', label: 'Watchlist', icon: Star      },
  { to: '/news',      label: 'News',      icon: Newspaper },
  { to: '/settings',  label: 'Settings',  icon: Settings  },
]

export default function Layout({ children }: { children: React.ReactNode }) {
  const { connected } = useWebSocket()

  return (
    <div className="flex h-screen bg-bg-900 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-52 flex-shrink-0 bg-bg-800 border-r border-bg-600 flex flex-col">
        {/* Logo */}
        <div className="px-4 py-5 border-b border-bg-600">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-accent" />
            <span className="font-bold text-white tracking-wide">Stocks</span>
          </div>
          <p className="text-xs text-gray-600 mt-0.5">Trading Platform</p>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2 py-4 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-accent/15 text-accent'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-bg-700'
                }`
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Connection status */}
        <div className="px-4 py-3 border-t border-bg-600">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-gain animate-pulse' : 'bg-gray-600'}`} />
            <span className="text-xs text-gray-500">{connected ? 'Live' : 'Polling'}</span>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto p-6">
          {children}
        </div>
      </main>
    </div>
  )
}
