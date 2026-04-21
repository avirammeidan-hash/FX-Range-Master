import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import StockDashboard from './pages/StockDashboard'
import Watchlist from './pages/Watchlist'
import NewsFeed from './pages/NewsFeed'
import Settings from './pages/Settings'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/"          element={<Navigate to="/stocks" replace />} />
        <Route path="/stocks"    element={<StockDashboard />} />
        <Route path="/watchlist" element={<Watchlist />} />
        <Route path="/news"      element={<NewsFeed />} />
        <Route path="/settings"  element={<Settings />} />
      </Routes>
    </Layout>
  )
}
