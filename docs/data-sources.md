# FX-Range-Master: Data Sources Reference

## Live Reference
- **Investing.com USD/ILS**: https://www.investing.com/currencies/usd-ils

## API Sources Comparison

| Source | Real-time? | Status | Key Required | Rate Limit | Notes |
|--------|-----------|--------|-------------|------------|-------|
| **Twelve Data** | Yes (real-time) | Active (primary) | Yes (free) | 800 calls/day, 8/min | Best free real-time option |
| **Yahoo Finance** | ~15min delay | Active (secondary) | No | Unlimited | yfinance library, good for candles |
| **open.er-api** | 1x/day | Active (fallback) | No | Unlimited | Daily rates only |
| **Currency API (fawazahmed0)** | ~hourly | Free, no key | No | Unlimited | GitHub-hosted, decent frequency |
| **Frankfurter (ECB)** | 1x/day | Free, no key | No | Unlimited | European Central Bank rates |
| **Alpha Vantage** | Real-time | Available | Yes (free) | 25 calls/day | Too limited for production |
| **Investing.com** | Real-time | Blocks API (403) | N/A | N/A | Web scraping blocked |
| **NetDania** | Real-time | No public API | N/A | N/A | No API access |

## Current Implementation

### Priority Order (in app.py `get_price_source`)
1. **Twelve Data** (real-time, API key: stored in config)
2. **Yahoo Finance** (yfinance, ~15min delay)
3. **open.er-api** (daily fallback)

### API Details

#### Twelve Data
- **Endpoint**: `https://api.twelvedata.com/price?symbol=USD/ILS&apikey=KEY`
- **Response**: `{"price": "3.12712"}`
- **Rate limit**: 8 calls/minute, 800/day (free tier)
- **Signup**: https://twelvedata.com (free, 10 seconds)
- **Key**: Set via `TWELVE_DATA_KEY` env var or config.yaml

#### Yahoo Finance
- **Library**: `yfinance` (Python)
- **Symbol**: `USDILS=X`
- **Delay**: ~15 minutes during market hours
- **Candles**: Best source for OHLC candle data (1m, 5m, 15m, 1h, 1d intervals)
- **Note**: Global forex hours (Sun 5pm ET - Fri 5pm ET), no data during Israeli-only hours

#### open.er-api
- **Endpoint**: `https://open.er-api.com/v6/latest/USD`
- **Response**: JSON with `rates.ILS` field
- **Updates**: Once daily at midnight UTC
- **No key required**

#### Currency API (fawazahmed0)
- **Endpoint**: `https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json`
- **Updates**: ~hourly
- **No key required**
- **Note**: Potential additional fallback

## Data Collection (Cloud Scheduler)
- **Endpoint**: `/api/collect?key=COLLECT_SECRET`
- **Frequency**: Every 2 minutes via Google Cloud Scheduler
- **Storage**: Firestore `price_history` collection
- **Dedup**: Skips storage when price unchanged from last collection
- **Record fields**: timestamp, price, source, baseline, upper, lower, position_pct, rsi, atr, gap_pct, ml_prediction, confidence, vix, news_sentiment

## Market Hours
- **Global Forex**: Sunday 5pm ET (22:00 UTC) to Friday 5pm ET
- **TASE (Tel Aviv)**: Sunday-Thursday, 9:45 AM - 5:30 PM Israel time
- **Best real-time coverage**: Twelve Data works during all forex hours
- **Gap**: Saturday (Shabbat) - no trading anywhere
