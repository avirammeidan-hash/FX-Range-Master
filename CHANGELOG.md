# Changelog

## v1.0.0 (2026-03-24)

### Features
- Full-screen responsive dashboard (3-column, 2-row CSS Grid layout)
- Real-time USD/ILS price tracking with 30-second refresh cycle
- AI Decision Engine with Random Forest model (TRADE/SKIP signals + confidence %)
- Speedometer-style confidence gauge with 5-segment color scale
- LIVE DATA indicator with uptime counter and connection status
- Multi-timeframe candlestick chart (10m, 20m, 30m, 1H, 1D, 5D, 1M, 3M)
- Candle hover with pip change and range display
- Price Position in Window with proximity-based gauges
- Key Levels table (Baseline, Upper, Lower, Stop Upper, Stop Lower)
- News Sentiment feed with expand mode, auto-refresh (2 min), and keyword highlighting
- Trading Signals panel with signal history log
- Market Context Bar (session, volatility, events, VIX)
- Simulation mode (pause live data, step through scenarios on real dashboard)
- Performance & Analytics modal with backtest statistics
- Trade suggestion engine with contextual recommendations

### Admin & Monitoring
- Admin panel with user management (create, delete users)
- Per-user activity monitoring (login count, events, data views, 30-day heatmap)
- Firebase Authentication with admin role management
- Cloud Firestore activity logging (login, data_refresh, simulation, performance)
- About modal with version info and changelog (admin only)

### Infrastructure
- Google Cloud Run deployment (me-west1, auto-scaling)
- Cloud Scheduler data collection every 2 minutes (skip-duplicate logic)
- Multiple data sources: Yahoo Finance (primary), ExchangeRate API (fallback)
- Firebase Auth + Firestore for user management and analytics
- Docker containerized with PYTHONUNBUFFERED for log visibility
- Firestore composite index for efficient activity queries

### Data & ML
- 15-year USD/ILS historical dataset (yfinance + Frankfurter API)
- Random Forest classifier trained on gap%, ATR, RSI, volatility, session, position
- Continuous price_history collection in Firestore for model retraining
