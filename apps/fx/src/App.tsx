import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ChartPage from './pages/ChartPage'
import NewsFeed from './pages/NewsFeed'
import Analysis from './pages/Analysis'
import Performance from './pages/Performance'
import History from './pages/History'
import Settings from './pages/Settings'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/"            element={<Dashboard />}   />
        <Route path="/chart"       element={<ChartPage />}   />
        <Route path="/news"        element={<NewsFeed />}    />
        <Route path="/analysis"    element={<Analysis />}    />
        <Route path="/performance" element={<Performance />} />
        <Route path="/history"     element={<History />}     />
        <Route path="/settings"    element={<Settings />}    />
      </Routes>
    </Layout>
  )
}
