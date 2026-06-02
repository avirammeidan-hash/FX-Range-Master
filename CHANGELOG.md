# Changelog

All notable changes to FX-Range-Master are documented here.
Format: `## vX.Y.Z (YYYY-MM-DD)` · Sections: Features · ML · UX · Infra · Fixes

---

## v1.8.3 (2026-06-02)

### Live data, less drift
- **New endpoint `/api/backtest-stats`** — returns model version, accuracy, age, feature importance (top 10), walk-forward backtest summary. Replaces hardcoded numbers in the dashboard.
- **KB Feature Importance bar chart** — now fetches live from `/api/backtest-stats` instead of hardcoded percentages. Macro features (`prev_vix_level`, `prev_dxy_return`, `prev_10y_yield`) are highlighted with a ✦ marker.
- **Test for new route** — `/api/backtest-stats` added to required-routes set.

---

## v1.8.2 (2026-06-02)

### Dev infrastructure
- **CI hardening (PR #2):** new `smoke` job runs `python -c "import app, auth, ml_filter, ..."` and asserts required Flask routes exist. Catches the `ImportError` class of bug that caused the v1.8.1 hotfix (`get_firestore` dropped from `auth.py`). Plus: `pull_request` trigger so all checks run on PRs (was push-only), and Node 24-compatible action versions.
- **Pytest scaffold (PR #3):** 19 baseline tests covering imports, routes, ML v3 features, admin config, and version drift. Runtime ~5s. Wired into the smoke CI job. Caught a real CHANGELOG ordering bug on first run.
- **Local pre-deploy script:** `scripts/local_check.ps1` — runs the same checks CI does, in ~10s locally. Run before every push.
- **Version unification (PR #4):** single source of truth — `VERSION` file at repo root. Flask context_processor injects `{{ app_version }}` into all templates. Bumping a release now means editing one file instead of four.

---

## v1.8.1 (2026-06-02) — Hotfix

### Fix
- Restore `get_firestore()` and `firestore` import in `auth.py` — accidentally dropped during merge, caused `ImportError` → `Service Unavailable` on startup
- Restored full v1.7.4 `index.html` (~1,600 lines) that a worktree partial copy had wiped, then re-applied only the intended v1.8.0 changes
- Made financial disclaimer footer `position: fixed` (was hidden by `body { overflow: hidden }`)
- Gated `Simulate`, `Admin`, `v1.8.1 (About)` buttons as `admin-only` so regular users don't see admin controls
- Redesigned `👁 User View` preview banner — subtle dark theme with admin email, no longer overlaps the header

---

## v1.8.0 (2026-06-02)

### UX — Copy & Empty States
- **Header badge** renamed: `AI-POWERED` → `USD/ILS RANGE INTELLIGENCE` for clearer product positioning
- **Login page subtitle** updated: `USD/ILS AI-Powered Trading Platform` → `USD/ILS Mean-Reversion Signal Engine`
- **Login badge** updated to match dashboard: `AI-POWERED ENGINE` → `USD/ILS RANGE INTELLIGENCE`
- **Empty / loading states** replaced with human-readable messages:
  - `Loading ML model...` → `Analyzing market conditions…`
  - `Loading params...` → `Connecting to live feed…`
  - `Loading candle data...` → `Fetching price history…`
  - `Loading news feed...` → `Fetching market news…`
  - `Waiting for signals...` → `Monitoring — no signals yet`
- **Financial disclaimer footer** added below dashboard:
  > *Educational tool · Not financial advice · Past performance and backtested results are not a guarantee of future returns. Live data via market feeds · Signals are for informational purposes only.*

### ML — v3 Feature Set (+3 external macro features)
External macro indicators added to the Random Forest skip-day filter, statistically validated over **5 years / 1,300 daily bars (2021–2026)**:

| Feature | Source | Validated Signal |
|---------|--------|-----------------|
| `prev_vix_level` | `^VIX` via yfinance | VIX ≥ 30 yesterday → mean USD/ILS +0.267% today; lag r=-0.130*** |
| `prev_dxy_return` | `DX-Y.NYB` via yfinance | DXY prev-day return; lag r=-0.126*** |
| `prev_10y_yield` | `^TNX` via yfinance | 10Y yield level context; same-day r=+0.305*** |

- Added `MLSkipFilter._fetch_external_indicators()` — fetches `^VIX`, `DX-Y.NYB`, `^TNX` at train time and at each daily prediction
- External data enriched into both `train()` and `predict_today()` pipelines
- Backward compatible: neutral fill values used if external data unavailable (VIX=20, DXY=0%, TNX=4%)
- `MODEL_VERSION` bumped `v2 → v3` — forces automatic retrain on next startup
- Total features: **16 → 19**

### Knowledge Base (my_guide.md)
- Added **"Signals & Market Correlations"** section with full empirical tables:
  - Same-day and lagged correlation table for S&P, VIX, DXY, 10Y yield, Gold, Nasdaq
  - VIX regime analysis (calm / normal / fear) with mean & std dev
  - S&P big-move impact on USD/ILS (same-day and next-day)
  - Future signals roadmap: CFTC COT, Fear & Greed Index, BoI intervention flag, FEDFUNDS

### Infrastructure
- **`firebase-service-account.json` added to `.gitignore`** — was missing, security fix
- **`auth.py`**: added `FIREBASE_SERVICE_ACCOUNT_JSON` environment variable support as fallback for Cloud Run deployments where the JSON file is not on disk. Priority: file → env var → bypass mode
- **GitHub secret `FIREBASE_SA_JSON`** set and cloud redeployment triggered — fixes admin panel showing "No users registered yet" (was in bypass mode due to missing secret)
- **Firebase authorized domain `127.0.0.1`** added — enables Google OAuth sign-in on local dev server

---

## v1.7.4 (2026-04-22)

### Fixes
- Gracefully handle invalid/empty Firebase SA JSON at startup (no crash on missing secret)
- Regenerate root `package-lock.json` after `apps/fx` version bump
- Add `stop_adaptive` to FxStatus params type — unblocks CI deploy

---

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
- Cloud Firestore activity logging
- About modal with version info and changelog (admin only)

### Infrastructure
- Google Cloud Run deployment (me-west1, auto-scaling)
- Cloud Scheduler data collection every 2 minutes
- Multiple data sources: Yahoo Finance (primary), ExchangeRate API (fallback)
- Firebase Auth + Firestore for user management and analytics
- Docker containerized

### Data & ML
- 15-year USD/ILS historical dataset
- Random Forest classifier (16 features: gap%, ATR, RSI, volatility, session, position)
- Continuous price_history collection in Firestore for model retraining
