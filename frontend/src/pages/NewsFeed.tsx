import { RefreshCw } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { refreshNews, getNews } from '../services/api'
import { NewsList } from '../components/NewsCard'
import { usePolling } from '../hooks/usePolling'

export default function NewsFeed() {
  const qc = useQueryClient()
  const { data, isLoading } = usePolling('news', getNews, 20000)

  const handleRefresh = async () => {
    await refreshNews()
    qc.invalidateQueries({ queryKey: ['news'] })
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-bold">News Feed</h1>
        <button className="btn-ghost flex items-center gap-1.5 text-xs" onClick={handleRefresh}>
          <RefreshCw className="w-3.5 h-3.5" /> Force refresh
        </button>
      </div>

      {isLoading ? (
        <div className="card text-center py-12 text-gray-500 text-sm">
          <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" /> Loading news...
        </div>
      ) : (
        <NewsList
          alerts={data?.alerts ?? []}
          sentiment={data?.sentiment ?? { sentiment: 'NEUTRAL', score: 0, alert_count: 0 }}
        />
      )}
    </div>
  )
}
