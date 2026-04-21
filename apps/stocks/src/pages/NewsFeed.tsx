import { Newspaper, RefreshCw } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { usePolling } from '@trading/hooks'
import { NewsList } from '@trading/ui'
import { getNews } from '../services/api'

export default function NewsFeed() {
  const qc = useQueryClient()
  const { data, isLoading } = usePolling('news-all', () => getNews(), 60_000)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Newspaper className="w-5 h-5 text-accent" />
          <h1 className="text-xl font-bold">Market News</h1>
        </div>
        <button
          onClick={() => qc.invalidateQueries({ queryKey: ['news-all'] })}
          className="btn-ghost flex items-center gap-1.5 text-xs"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {isLoading ? (
        <div className="card text-center py-12 text-gray-500">
          <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" /> Loading…
        </div>
      ) : (
        <NewsList
          articles={data?.articles ?? []}
          sentiment={data?.sentiment ?? { sentiment: 'NEUTRAL', score: 0, alert_count: 0 }}
          maxHeight={600}
        />
      )}
    </div>
  )
}
