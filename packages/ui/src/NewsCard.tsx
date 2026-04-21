import { ExternalLink, AlertTriangle, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import type { NewsArticle, SentimentSummary } from '@trading/api-client'

interface NewsListProps {
  articles: NewsArticle[]
  sentiment: SentimentSummary
  maxHeight?: number
}

/**
 * News sentiment header + scrollable article list.
 * Works for FX news, stock-specific news, or market-wide news.
 */
export function NewsList({ articles, sentiment, maxHeight = 256 }: NewsListProps) {
  const sentColor =
    sentiment.sentiment === 'BULLISH' ? 'text-gain' :
    sentiment.sentiment === 'BEARISH' ? 'text-loss' : 'text-gray-400'

  const SentIcon =
    sentiment.sentiment === 'BULLISH' ? TrendingUp  :
    sentiment.sentiment === 'BEARISH' ? TrendingDown : Minus

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">News Sentiment</h3>
        <div className={`flex items-center gap-1 text-sm font-bold ${sentColor}`}>
          <SentIcon className="w-4 h-4" />
          {sentiment.sentiment}
          <span className="text-xs font-normal text-gray-500 ml-1">
            ({sentiment.score >= 0 ? '+' : ''}{sentiment.score.toFixed(2)})
          </span>
        </div>
      </div>

      {sentiment.alert_count > 0 && (
        <div className="flex items-center gap-1.5 text-xs text-warn">
          <AlertTriangle className="w-3.5 h-3.5" />
          {sentiment.alert_count} active alert{sentiment.alert_count !== 1 ? 's' : ''}
        </div>
      )}

      {articles.length === 0 ? (
        <p className="text-sm text-gray-600 text-center py-4">No recent news</p>
      ) : (
        <div className="space-y-2 overflow-y-auto no-scrollbar" style={{ maxHeight }}>
          {articles.map((a, i) => <NewsItem key={i} article={a} />)}
        </div>
      )}
    </div>
  )
}

function NewsItem({ article }: { article: NewsArticle }) {
  const impact = article.impact ?? 0
  const impactColor =
    impact >= 0.7 ? 'bg-loss'   :
    impact >= 0.4 ? 'bg-warn'   :
    impact >= 0.1 ? 'bg-accent' : 'bg-gray-700'

  const ts = article.timestamp
    ? formatDistanceToNow(new Date(article.timestamp), { addSuffix: true })
    : ''

  return (
    <div className="bg-bg-700 rounded-lg px-3 py-2 space-y-1">
      <div className="flex items-start gap-2">
        <span className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 ${impactColor}`} />
        <div className="min-w-0 flex-1">
          <p className="text-xs text-gray-200 leading-snug line-clamp-2">{article.title}</p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs text-gray-500">{article.source}</span>
            {ts && <span className="text-xs text-gray-600">{ts}</span>}
            {article.url && (
              <a href={article.url} target="_blank" rel="noopener noreferrer"
                className="ml-auto text-accent hover:text-accent-400 transition-colors">
                <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>
          {article.summary && (
            <p className="text-xs text-gray-500 mt-1 line-clamp-2">{article.summary}</p>
          )}
        </div>
      </div>
      {article.keywords && article.keywords.length > 0 && (
        <div className="flex flex-wrap gap-1 pl-3.5">
          {article.keywords.slice(0, 4).map((kw, j) => (
            <span key={j} className="text-xs px-1.5 py-0.5 rounded bg-bg-600 text-gray-400">{kw}</span>
          ))}
        </div>
      )}
    </div>
  )
}

export default NewsList
