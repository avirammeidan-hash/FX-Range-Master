import { Settings as SettingsIcon } from 'lucide-react'

export default function Settings() {
  return (
    <div className="space-y-4 max-w-lg">
      <div className="flex items-center gap-2">
        <SettingsIcon className="w-5 h-5 text-accent" />
        <h1 className="text-xl font-bold">Settings</h1>
      </div>
      <div className="card space-y-4">
        <div>
          <label className="text-sm text-gray-400">Backend URL</label>
          <input
            defaultValue="http://localhost:5001"
            className="mt-1 w-full px-3 py-2 bg-bg-800 border border-bg-600 rounded-lg text-sm text-gray-200 focus:outline-none focus:border-accent"
          />
        </div>
        <div>
          <label className="text-sm text-gray-400">Default ticker</label>
          <input
            defaultValue="AAPL"
            className="mt-1 w-full px-3 py-2 bg-bg-800 border border-bg-600 rounded-lg text-sm text-gray-200 focus:outline-none focus:border-accent"
          />
        </div>
        <div>
          <label className="text-sm text-gray-400">Refresh interval (seconds)</label>
          <input
            type="number"
            defaultValue={30}
            className="mt-1 w-full px-3 py-2 bg-bg-800 border border-bg-600 rounded-lg text-sm text-gray-200 focus:outline-none focus:border-accent"
          />
        </div>
      </div>
    </div>
  )
}
