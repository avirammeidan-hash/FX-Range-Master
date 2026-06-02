# My Design & UX Guidance -- FX-Range-Master

Summary of all design decisions and guidance provided during development.

> **Release Notes Policy:** Every code change, feature, or fix must be documented in **both**:
> 1. `CHANGELOG.md` — versioned entry under the correct `vX.Y.Z` section
> 2. This KB (`my_guide.md`) — in the relevant section (UX, ML, Infra, Signals, etc.)
>
> Version scheme: `MAJOR.MINOR.PATCH` — bump MINOR for features, PATCH for fixes.
> Current version: **v1.8.0** (2026-06-02)

---

## 1. Full-Screen Layout (No Scrolling)

**Request:** "is there a reason why the UI isn't scalable with use the whole screen or page layout the page layout should present all the important data without scrolling up/down left right"

**Decision:** Dashboard must fill 100vh viewport. No scrolling on desktop. Use CSS Grid with named areas to distribute all panels across the full screen. Responsive breakpoints allow scrolling only on tablet (1200px) and mobile (768px).

---

## 2. Guide Modal -- Full Page with Close Option

**Request:** "for the guide clip also take all page layout with closed option"

**Decision:** The trading guide opens as a full-page overlay (`position: fixed; inset: 0`) covering the entire viewport, not a small centered popup. Close button in the header. Background is fully opaque (not transparent overlay).

---

## 3. Eliminate Empty Space

**Request:** "why there is an empty space in the middle of the page?"

**Decision:** Removed the dedicated "action" grid area that sat empty when no signal was active. Action banner is now a floating overlay toast that appears on top of content when triggered. Every grid cell shows useful data -- no reserved empty space.

---

## 4. Bigger / Vivid Numbers

**Request:** "important text number should be bigger/vivid"

**Decision:** Price hero = 64px white bold. Daily change = 22px green/red. Distance to bounds = 32px vivid red/green. Key Level prices = 18px bold. All numbers use `font-variant-numeric: tabular-nums` for alignment. Color = semantic meaning only (green=buy/profit, red=sell/loss, cyan=baseline).

---

## 5. Reference Trading Platform Design Style

**Request:** "check from web a recommended trade platform UI for reference design"

**Decision:** Design inspired by:
- **TradingView** -- Dark theme, blue accent, information density, zero wasted space
- **Binance** -- Order book depth bars, large spread price, dense panels
- **Bloomberg Terminal** -- Monospace alignment, conceal complexity, surface critical metrics
- **thinkorswim** -- Flexible grid, active trader view
- **Dribbble concepts** -- Card-based, hero numbers 28-48px, dark blue backgrounds

---

## 6. Center-Aligned Numbers

**Request:** "the numbers should be middle area align"

**Decision:** All metric numbers center-aligned within their cards/cells. This follows the modern dashboard card style (Dribbble trend) rather than the traditional right-aligned financial terminal style.

---

## 7. Match Reference Design Style (Professional Trading Platform)

**Request:** (shared reference mockup image) "design in this style"

**Decisions:**
- **Card titles** = 16px bold white (not small muted uppercase)
- **Cards** = 14px border-radius, 22px padding, generous spacing
- **Key Levels** = Table format with Level / Price / Status columns
- **Status indicators** = Dynamic tags: "Active" (cyan), "Normal" (green), "Approaching" (yellow), "Triggered" (red)
- **Semi-circular gauge dials** for distance meters (SVG arcs, 14px thick stroke, 160px wide)
- **Country flag images** next to news items (rectangular flags from flagcdn.com, not emoji)
- **Signal badges** = Bigger pills with border outline (6px 18px padding, 8px radius)

---

## 8. Replace Cryptic Labels with Visual Support Indicators

**Request:** "near the flag why there is usd usd+ usd+!! what does it say? do we need it? i suggest if event supporting your suggestion to sell or buy marked it by the level of supporting"

**Decision:** Replaced confusing "USD+ !!" text labels with visual support level bars:
- 1-4 ascending bars showing strength of signal support
- **Green bars** = news supports BUY direction
- **Red bars** = news supports SELL direction
- **Gray bars** = neutral news
- Small "Buy" / "Sell" label beneath the bars
- Bar count scales with news score magnitude (abs score 1=2 bars, 2=3, 3+=4)

---

## 9. Real-Time News Monitoring (Including Trump / Truth Social)

**Request:** "can you get realtime events such as Trump network the truth?"

**Decision:** Built RSS-based news monitor polling Reuters, CNBC, MarketWatch, ForexLive every 5 minutes. Major outlets pick up Trump's Truth Social posts within minutes, so RSS is sufficient without needing direct Truth Social API. 50+ keyword scoring rules tuned for USD/ILS impact. Optional NewsAPI.org integration for deeper coverage.

---

## 10. Configurable Parameters (Not Hardcoded)

**Request:** "if i want to change the 1% window size? will it improve the success? what is the optimal window size and can it be a configured parameter"

**Decision:** All trading parameters must be configurable from `config.yaml`, not hardcoded in source code. Window width (half_width_pct) and stop loss extension (stop_loss_extension_pct) are read from config at startup. Dashboard dynamically shows current params. Optimizer proved W=0.3%, S=0.8% is optimal across 49 combinations.

---

## 11. Interactive Trading Simulation for User Education

**Request:** "show me a clip illustration to understand when you tell me to buy or sell and the supported index level/confidence score add this to the html for the user help assistance"

**Decision:** Built an 8-step animated trading simulation inside the Guide modal. Shows a simulated Sunday session with SVG price chart, animated price line, signal markers (buy/sell/TP triangles), mini dashboard (price/position/P&L), and narration explaining each step. Auto-plays at 4-second intervals. Includes a 5-factor confidence score meter (Event Calendar, VIX, News, S&P/NASDAQ, Win Rate) with animated fill bars.

---

## 12. Platform Performance Section for Professional Credibility

**Request:** "add to the clip something like Platform Performance for marketing the platform that professional user will understand something like Day Type Breakdown"

**Decision:** Added comprehensive backtested results section to the Guide:
- **Hero stats** -- 73% WR, 2.55 PF, +6.90 P&L, -0.28 MaxDD
- **Day Type Breakdown** table -- 8 rows, 780 trading days (Normal, OPEX, Month-End, Event, High-Impact, High-Volume, S&P Down, Volatile Spike)
- **Directional Performance** -- SHORT 70.7% vs LONG 60.4% cards
- **Market Indicator Correlations** -- S&P, NASDAQ, Oil, VIX, Buy/Sell Pressure
- **Event-Specific USD/ILS Bias** -- 6 event cards (JOLTS, GDP, FOMC Min, BOI, Retail, IL CPI)
- **Optimal Skip Filter** recommendation
- **Methodology note** -- walk-forward, no look-ahead bias

---

## 13. Data Must Be Verifiable

**Request:** "where are these numbers Event Correlation Findings (3 years) Day Type Breakdown?"

**Decision:** All statistics shown in the platform must come from actual backtesting/analysis output, not invented. Numbers were verified against analyzer.py and optimizer.py output. If data can't be sourced from real analysis, don't show it.

---

## 14. Keep Iterating Until Quality Matches Reference

**Request:** "still this design view is better (the country flags) the text the images all is better try again"

**Decision:** Don't settle for "close enough." When given a reference design, iterate until the visual quality truly matches -- flag images (not emoji), text sizing, spacing, card styling. Each iteration should address specific gaps identified by comparing screenshot vs reference.

---

## General Principles Established

1. **Every pixel shows data** -- no decorative whitespace or empty panels
2. **Color = meaning** -- green (buy/profit/good), red (sell/loss/bad), cyan (baseline/info), yellow (caution), muted gray (secondary)
3. **Visual hierarchy** -- biggest number = most important (price > change > distances > levels)
4. **Pro trading feel** -- dark theme, dense information, tabular numbers, semantic colors
5. **Instant readability** -- trader glances at dashboard and understands market state in 2 seconds
6. **Visual indicators over text** -- bars/gauges/tags instead of raw text labels
7. **Configurable over hardcoded** -- trading parameters in config.yaml, not source code
8. **Data-driven decisions** -- show only verifiable backtested statistics
9. **Professional audience** -- design and content should impress experienced traders, not beginners
10. **Self-explanatory UI** -- interactive simulation + visual guide so user never needs a manual

---

# Data Sources & AI/ML Knowledge Base

## Data Sources

| Source | Type | Size | Period | File |
|--------|------|------|--------|------|
| Yahoo Finance (yfinance) | Daily OHLC | 2,918 bars | 2015-01-01 to 2026-03-20 | `usd_ils_daily_10y.csv` |
| Yahoo Finance (yfinance) | Hourly OHLC | 12,007 bars | 2024-03-20 to 2026-03-20 | `usd_ils_hourly_2y.csv` |
| Yahoo Finance (yfinance) | 1-min OHLC | ~28 days | Last 28 days (rolling) | Fetched live via `simulator.py` |
| Yahoo Finance (^VIX) | Daily VIX | Real-time | Current day | Fetched live |
| RSS feeds (Reuters, CNBC, MarketWatch, ForexLive) | News headlines | ~50 items/poll | 5-min polling | `news_monitor.py` |
| NewsAPI.org (optional) | News articles | 100 req/day free | Real-time | API key in `config.yaml` |
| Events calendar | Economic events | 16 event types | 2023-2026 | Hardcoded in `events.py` |

### How to Get More Data
- **Investing.com CSV**: Download 10+ year hourly data (free). Simulator already supports Investing.com format via `_clean_investing_csv()`.
- **GitHub repos**: `ben-dom/forex-historical-data` (H1 data), `ejtraderLabs/historical-data` (10yr H1).
- **Kaggle**: `alifougi/forex-currency-pairs-dataset-in-1-hour-timeframe` (60,000+ rows of H1 forex data).
- **Multi-pair expansion**: Test strategy on USD/TRY, USD/ZAR, USD/MXN, EUR/ILS, USD/PLN to multiply training data.

## AI/ML Algorithms Tested

### Filters Tested (filter_backtest.py -- 2y hourly data)

| Filter | Trades | WR | PF | P&L | Verdict |
|--------|--------|-----|------|-----|---------|
| **BASELINE (current)** | 1,693 | 73.0% | 2.55 | +6.90 | **Best on hourly** |
| ATR-Adaptive bands | 187 | 10.2% | inf | +0.46 | Too few trades |
| Confirmation bar | 971 | 59.4% | 1.36 | +0.02 | Worse |
| Time-of-day filter | All day best | -- | -- | -- | No improvement |
| Kalman-smoothed baseline | 1,698 | 73.0% | 2.56 | +6.95 | ~Identical |
| All combined | 166 | 9.0% | inf | +0.01 | Killed trade count |

**Verdict**: No algorithmic filter improved the core strategy on hourly data. Simplicity is the edge.

### ML Skip-Day Classifier (ml_backtest.py -- 10y daily data)

| Model | Trades | WR | PF | P&L |
|-------|--------|-----|------|-----|
| Baseline (no ML) | 2,907 | 44.1% | 0.72 | -7.40 |
| **Random Forest (train=250d)** | **1,350** | **58.4%** | **2.54** | **+6.57** |
| RF @ 65% confidence | 1,058 | 59.5% | 3.47 | +6.26 |
| RF @ 80% confidence | 688 | 59.3% | 6.00 | +4.80 |
| Gradient Boosting (train=250d) | 1,344 | 57.8% | 2.24 | +5.92 |

**Winner**: Random Forest with 250-day rolling training window.
**Accuracy**: 81.2% (walk-forward, no look-ahead bias).

### ML Feature Importance (Top 5)

| Feature | Importance | Description |
|---------|-----------|-------------|
| **abs_gap_pct** | 40.1% | Absolute overnight gap size -- dominant predictor |
| gap_pct | 15.3% | Directional overnight gap |
| prev_range_pct | 13.1% | Yesterday's high-low range as % |
| atr5_pct | 6.7% | 5-day ATR (short-term volatility) |
| prev_return_pct | 5.0% | Yesterday's close-to-close return |

**Key Insight**: Large overnight gaps predict bad mean-reversion days. The ML model learned to skip those days.

### Why NOT Neural Networks

1. **Data size**: ~2,900 daily bars is too small. NNs need 10,000+ samples minimum.
2. **Structural edge**: USD/ILS mean-reverts due to BoI intervention patterns -- a fixed market property, not a learnable pattern.
3. **Overfitting risk**: Tree models (RF, XGBoost) handle small datasets better and are interpretable.
4. **Noise**: Currency data is extremely noisy. NNs amplify noise with insufficient data.

### ML Integration (ml_filter.py)

The Random Forest skip-day filter is integrated into the live system:
- **Training**: Auto-trains on startup from `usd_ils_daily_10y.csv`, saves model to `ml_model.pkl`. Retrains weekly.
- **Prediction**: At market open, computes 16 features from recent 60 days of data, predicts trade/skip.
- **Threshold**: Default 60% confidence. Below threshold = skip day. Configurable.
- **Dashboard**: Shows ML trade/skip decision with confidence score in both CLI and web UI.
- **Fallback**: If ML unavailable (data error, model not trained), defaults to TRADE (no skip).

---

## Signals & Market Correlations (Empirical — 5 Years Daily Data, 2021–2026)

Statistical analysis of 1,300 USD/ILS trading days vs. major external indicators.
All correlations computed via Pearson r on daily returns unless noted.

### Same-Day Correlations with USD/ILS

| Indicator | Ticker | r (same day) | Significance | Interpretation |
|-----------|--------|-------------|--------------|----------------|
| **S&P 500** | ^GSPC | **-0.390** | *** (p<0.001) | S&P rises → shekel strengthens (risk-on) |
| **10Y Treasury Yield** | ^TNX | **+0.305** | *** | Higher yields → USD demand rises |
| **VIX** | ^VIX | -0.239 | *** | VIX daily return vs USD/ILS (see regime below) |
| **DXY (Dollar Index)** | DX-Y.NYB | -0.210 | *** | Global dollar strength correlated |
| **Gold** | GC=F | +0.131 | *** | Safe-haven flows |
| Nasdaq | ^NDX | -0.023 | ✗ (p=0.40) | Not significant alone |

### Lagged Correlations (Yesterday → Today USD/ILS)

| Indicator | r (t-1 → t) | Significance | Use in ML |
|-----------|-------------|--------------|-----------|
| **VIX level** | -0.130 | *** | ✅ Added as `prev_vix_level` |
| **DXY return** | -0.126 | *** | ✅ Added as `prev_dxy_return` |
| S&P 500 return | -0.022 | ✗ | ❌ Not predictive |
| Nasdaq return | -0.029 | ✗ | ❌ Not predictive |

### VIX Regime Analysis (by VIX level, not return)

| VIX Level | Days (5yr) | USD/ILS Mean | Std Dev | Implication |
|-----------|-----------|--------------|---------|-------------|
| **< 15** (calm) | 276 | **+0.20%** | 0.57% | Narrow range — strategy works well |
| **15–25** (normal) | 840 | **+0.09%** | 0.89% | Standard conditions |
| **≥ 25** (fear) | 184 | **-0.33%** | 1.86% | ⚠️ 3× wider swings — risk both ways |

**Key insight:** When VIX ≥ 25, daily volatility triples. The mean-reversion window may be too narrow. Consider widening `half_width_pct` or skipping entirely.

### VIX Level Yesterday → USD/ILS Directional Bias Today

| VIX Yesterday | USD/ILS Today (mean) | n |
|---------------|---------------------|---|
| VIX ≥ 15 | +0.053% | 1,023 |
| VIX ≥ 20 | +0.083% | 439 |
| VIX ≥ 25 | +0.122% | 184 |
| **VIX ≥ 30** | **+0.267%** | 66 |

### S&P Big-Move Days (USD/ILS Reaction)

| S&P Move | Same-Day USD/ILS | Next-Day USD/ILS | n |
|----------|-----------------|-----------------|---|
| S&P < -1.5% | **+0.39%** | +0.03% | 86 |
| S&P flat (±0.5%) | +0.05% | +0.05% | 813 |
| S&P > +1.5% | **-0.36%** | -0.07% | 101 |

**Note:** The next-day effect is near zero — S&P is a same-day signal only, not predictive for next morning.

### ML Feature Additions (v3)

Three external macro features added to the Random Forest model (all use t-1 lag to avoid lookahead):

```
prev_vix_level   — VIX closing level yesterday      (neutral fill: 20.0)
prev_dxy_return  — DXY % return yesterday           (neutral fill: 0.0%)
prev_10y_yield   — 10Y Treasury yield level yesterday (neutral fill: 4.0%)
```

Sources fetched via yfinance at train time and at each prediction: `^VIX`, `DX-Y.NYB`, `^TNX`.

### Future Signals to Consider

| Signal | Source | Rationale |
|--------|--------|-----------|
| CFTC COT — ILS futures positioning | CFTC (weekly, free) | Institutional long/short on ILS; unique leading indicator |
| Fear & Greed Index | Alternative.me API (free) | 0–100 composite; extreme fear historically volatile for ILS |
| BoI Intervention Flag | Bank of Israel RSS / calendar | BoI buys USD to weaken shekel — overrides all signals |
| FEDFUNDS rate change | FRED API (free) | Fed hike/cut days: large immediate USD/ILS moves |
| HMM Regime on USD/ILS itself | Computed | Range vs trend detection for strategy validity |

---

## Real-Time Data Collection Plan (Sunday Morning Workflow)

### Pre-Market Setup (Sunday 07:00-07:50 Israel Time)

**Step 1: Refresh Historical Data (weekly)**
```bash
python -c "
import yfinance as yf
data = yf.download('ILS=X', start='2015-01-01', interval='1d')
data.to_csv('usd_ils_daily_10y.csv')
print(f'Updated: {len(data)} daily bars')
"
```

**Step 2: Retrain ML Model**
```bash
python -c "
from ml_filter import get_ml_filter
ml = get_ml_filter()
result = ml.train(retrain=True)
print(result)
"
```

**Step 3: Get Today's ML Prediction**
```bash
python -c "
from ml_filter import get_ml_filter
ml = get_ml_filter()
ml.train()
pred = ml.predict_today()
print(f'Trade: {pred[\"trade\"]} | Confidence: {pred[\"confidence\"]:.0%}')
print(f'Reason: {pred[\"reason\"]}')
if pred.get('features'):
    print(f'Gap: {pred[\"features\"][\"abs_gap_pct\"]:.4f}%')
"
```

**Step 4: Launch Dashboard**
```bash
python app.py
# Dashboard shows: ML TRADE/SKIP + confidence + events + VIX
```

### Real-Time Data Feeds During Trading

| Data | Source | Frequency | How |
|------|--------|-----------|-----|
| USD/ILS price | yfinance 1-min bars | Every 60s | `scanner.get_current_price()` |
| VIX level | yfinance ^VIX daily | At startup | `yf.Ticker("^VIX").history()` |
| News headlines | RSS feeds | Every 5 min | `news_monitor.py` polling |
| Event calendar | events.py | At startup | Static calendar lookup |
| ML prediction | ml_filter.py | At startup (daily) | Random Forest inference |

### Future Data Enhancements to Consider

1. **BoI intervention announcements** -- Scrape Bank of Israel press releases for real-time intervention alerts
2. **Shekel bond auction results** -- MOF auction data affects ILS supply/demand
3. **TASE index data** -- TA-35 correlation with USD/ILS
4. **Order flow / positioning** -- CFTC COT reports (weekly) for institutional ILS positioning
5. **Intraday VIX** -- Switch from daily to 1-min VIX for more responsive volatility filter
6. **Sentiment from X/Twitter** -- Track forex influencer sentiment on USD/ILS (API required)
