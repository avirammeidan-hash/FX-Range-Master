"""Retrain ML model on merged yfinance + Frankfurter dataset (15 years)."""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
warnings.filterwarnings('ignore')

# ── Load and merge both data sources ──────────────────────────────────────
print('=== LOADING DATA ===')

# 1. yfinance daily (has OHLC)
yf_df = pd.read_csv('usd_ils_daily_10y.csv', skiprows=[1, 2])
yf_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
yf_df['Date'] = pd.to_datetime(yf_df['Date'])
yf_df.set_index('Date', inplace=True)
for c in ['Close', 'High', 'Low', 'Open']:
    yf_df[c] = pd.to_numeric(yf_df[c], errors='coerce')
yf_df = yf_df.dropna(subset=['Close', 'High', 'Low', 'Open'])
yf_df.sort_index(inplace=True)
print(f'yfinance: {len(yf_df)} bars ({yf_df.index[0].date()} to {yf_df.index[-1].date()})')

# 2. Frankfurter (Close only — synthesize OHLC)
ff_df = pd.read_csv('usd_ils_frankfurter_26y.csv', index_col=0, parse_dates=True)
ff_df.columns = ['Close']
ff_df.sort_index(inplace=True)
print(f'Frankfurter: {len(ff_df)} bars ({ff_df.index[0].date()} to {ff_df.index[-1].date()})')

ff_df['Open'] = ff_df['Close'].shift(1)
ff_df['High'] = ff_df[['Open', 'Close']].max(axis=1) * 1.002
ff_df['Low'] = ff_df[['Open', 'Close']].min(axis=1) * 0.998
ff_df = ff_df.dropna()

# Merge: yfinance where available (real OHLC), Frankfurter for older dates
ff_only = ff_df[~ff_df.index.normalize().isin(yf_df.index.normalize())]
merged = pd.concat([ff_only[['Open', 'High', 'Low', 'Close']], yf_df[['Open', 'High', 'Low', 'Close']]])
merged.sort_index(inplace=True)
merged = merged[~merged.index.duplicated(keep='last')]
print(f'Merged: {len(merged)} unique bars ({merged.index[0].date()} to {merged.index[-1].date()})')

# ── Feature Engineering ───────────────────────────────────────────────────
print('\n=== BUILDING FEATURES ===')
close = merged['Close']
high = merged['High']
low = merged['Low']
open_ = merged['Open']

feat = pd.DataFrame(index=merged.index)
feat['gap_pct'] = ((open_ - close.shift(1)) / close.shift(1) * 100)
feat['abs_gap_pct'] = feat['gap_pct'].abs()
feat['prev_range_pct'] = ((high.shift(1) - low.shift(1)) / close.shift(1) * 100)
feat['prev_return_pct'] = close.pct_change().shift(1) * 100

tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
feat['atr5_pct'] = (tr.rolling(5).mean() / close * 100)
feat['atr14_pct'] = (tr.rolling(14).mean() / close * 100)
feat['vol_ratio'] = feat['atr5_pct'] / feat['atr14_pct']

feat['day_of_week'] = merged.index.dayofweek
feat['month'] = merged.index.month
feat['is_monday'] = (merged.index.dayofweek == 0).astype(int)
feat['is_friday'] = (merged.index.dayofweek == 4).astype(int)

sma20 = close.rolling(20).mean()
feat['dist_sma20_pct'] = ((close - sma20) / sma20 * 100)
feat['bb_width_pct'] = (close.rolling(20).std() / close * 100)

delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss_val
feat['rsi14'] = 100 - (100 / (1 + rs))
feat['momentum_5d'] = close.pct_change(5).shift(1) * 100

big_move = (feat['prev_range_pct'] > 1.0).astype(int)
feat['days_since_big_move'] = big_move.groupby((big_move != big_move.shift()).cumsum()).cumcount()

FEATURE_COLS = [
    'gap_pct', 'prev_range_pct', 'prev_return_pct', 'atr5_pct', 'atr14_pct',
    'vol_ratio', 'day_of_week', 'dist_sma20_pct', 'bb_width_pct', 'rsi14',
    'days_since_big_move', 'month', 'is_monday', 'is_friday', 'momentum_5d', 'abs_gap_pct',
]

# ── Label days ────────────────────────────────────────────────────────────
print('Labeling days...')
hw = 0.3 / 100.0
se = 0.8 / 100.0
labels = {}
for i in range(1, len(merged)):
    day = merged.index[i]
    baseline = float(merged['Close'].iloc[i - 1])
    bar = merged.iloc[i]
    upper = baseline * (1 + hw)
    lower = baseline * (1 - hw)
    stop_upper = baseline * (1 + hw + se)
    stop_lower = baseline * (1 - hw - se)
    h, l, c = bar['High'], bar['Low'], bar['Close']

    pnl = 0.0
    had_trade = False
    if h >= upper:
        had_trade = True
        if h >= stop_upper:
            pnl += upper - stop_upper
        elif l <= baseline:
            pnl += upper - baseline
        else:
            pnl += upper - c
    if l <= lower:
        had_trade = True
        if l <= stop_lower:
            pnl += stop_lower - lower
        elif h >= baseline:
            pnl += baseline - lower
        else:
            pnl += c - lower
    labels[day] = 1 if pnl > 0 else (0 if had_trade else -1)

labels_s = pd.Series(labels)

# ── Walk-forward backtest ─────────────────────────────────────────────────
print('\n=== WALK-FORWARD BACKTEST ===')
valid_idx = feat.dropna().index.intersection(labels_s.index)
valid_idx = valid_idx[labels_s.loc[valid_idx] >= 0]

X_all = feat.loc[valid_idx][FEATURE_COLS].values
y_all = labels_s.loc[valid_idx].values
dates_all = valid_idx

print(f'Total tradeable days: {len(X_all)}')

TRAIN_WINDOW = 250
results = []
model = None

for i in range(TRAIN_WINDOW, len(X_all)):
    X_train = X_all[i - TRAIN_WINDOW:i]
    y_train = y_all[i - TRAIN_WINDOW:i]
    X_test = X_all[i:i + 1]
    y_test = y_all[i]

    if model is None or i % 250 == 0:
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[0]
    conf = float(proba[1]) if len(proba) > 1 else float(proba[0])

    day = dates_all[i]
    idx = merged.index.get_loc(day)
    baseline = float(merged['Close'].iloc[idx - 1])
    bar = merged.iloc[idx]
    upper = baseline * (1 + hw)
    lower = baseline * (1 - hw)
    stop_upper = baseline * (1 + hw + se)
    stop_lower = baseline * (1 - hw - se)
    h, l, c = bar['High'], bar['Low'], bar['Close']

    day_pnl = 0.0
    trades = 0
    if h >= upper:
        trades += 1
        if h >= stop_upper:
            day_pnl += upper - stop_upper
        elif l <= baseline:
            day_pnl += upper - baseline
        else:
            day_pnl += upper - c
    if l <= lower:
        trades += 1
        if l <= stop_lower:
            day_pnl += stop_lower - lower
        elif h >= baseline:
            day_pnl += baseline - lower
        else:
            day_pnl += c - lower

    results.append({
        'date': day, 'pnl': day_pnl, 'trades': trades,
        'conf': conf, 'actual': y_test, 'predicted': 1 if conf >= 0.6 else 0
    })

rdf = pd.DataFrame(results)

# ── Results ───────────────────────────────────────────────────────────────
print(f'\n{"=" * 60}')
print(f'MERGED DATASET: {len(merged)} bars, {len(rdf)} OOS days')
print(f'{"=" * 60}')

# Baseline (no ML)
base = rdf[rdf['trades'] > 0]
base_wins = (base['pnl'] > 0).sum()
base_pnl = base['pnl'].sum()
base_wr = base_wins / len(base) * 100 if len(base) > 0 else 0
gross_win = base[base['pnl'] > 0]['pnl'].sum()
gross_loss = abs(base[base['pnl'] <= 0]['pnl'].sum())
base_pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

print(f'\nBASELINE (no ML):')
print(f'  Trades: {len(base)}, WR: {base_wr:.1f}%, PF: {base_pf:.2f}, PnL: {base_pnl:+.2f}')

# ML filtered at various thresholds
print(f'\nML FILTERED:')
for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]:
    ml = rdf[(rdf['conf'] >= thresh) & (rdf['trades'] > 0)]
    if len(ml) == 0:
        continue
    ml_wins = (ml['pnl'] > 0).sum()
    ml_wr = ml_wins / len(ml) * 100
    ml_pnl = ml['pnl'].sum()
    gw = ml[ml['pnl'] > 0]['pnl'].sum()
    gl = abs(ml[ml['pnl'] <= 0]['pnl'].sum())
    ml_pf = gw / gl if gl > 0 else float('inf')
    marker = ' <<<' if thresh == 0.6 else ''
    print(f'  >= {thresh:.0%}: Trades={len(ml):,}, WR={ml_wr:.1f}%, PF={ml_pf:.2f}, PnL={ml_pnl:+.2f}{marker}')

# Feature importance
print(f'\nTop Features:')
importances = dict(zip(FEATURE_COLS, model.feature_importances_))
for feat_name, imp in sorted(importances.items(), key=lambda x: -x[1])[:5]:
    print(f'  {feat_name}: {imp:.1%}')

# Accuracy
correct = (rdf['predicted'] == rdf['actual']).sum()
print(f'\nModel accuracy: {correct / len(rdf) * 100:.1f}%')

# ── Save retrained model ─────────────────────────────────────────────────
# Train final model on last 250 days for live use
from datetime import datetime
final_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, min_samples_leaf=10,
    random_state=42, n_jobs=-1
)
final_model.fit(X_all[-250:], y_all[-250:])
final_acc = round(float(np.mean(final_model.predict(X_all[-250:]) == y_all[-250:])) * 100, 1)

with open('ml_model.pkl', 'wb') as f:
    pickle.dump({
        'model': final_model,
        'train_date': datetime.now(),
        'feature_importance': {c: round(float(v), 4) for c, v in zip(FEATURE_COLS, final_model.feature_importances_)},
        'accuracy': final_acc,
    }, f)

print(f'\nFinal model saved to ml_model.pkl (accuracy: {final_acc}%)')
print(f'Trained on merged dataset: {len(merged)} total bars')
