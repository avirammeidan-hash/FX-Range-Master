# Session Notes & Lessons Learned

Cumulative log of session outcomes, lessons, and parked work — for use at the start of future sessions.

---

## Session: 2026-06-01 → 2026-06-02 — v1.8.0 / v1.8.1 release

### Live URL
**https://fx-range-master-403186329512.me-west1.run.app** (v1.8.1)

### What shipped (11 commits to main)

#### UX
- Badge: `AI-POWERED` → `USD/ILS RANGE INTELLIGENCE` (dashboard + login)
- Login subtitle: `USD/ILS Mean-Reversion Signal Engine`
- 5 empty/loading states humanised:
  - `Loading ML model...` → `Analyzing market conditions…`
  - `Loading params...` → `Connecting to live feed…`
  - `Loading candle data...` → `Fetching price history…`
  - `Loading news feed...` → `Fetching market news…`
  - `Waiting for signals...` → `Monitoring — no signals yet`
- Financial disclaimer footer (fixed bottom):
  > *Educational tool · Not financial advice · Past performance and backtested results are not a guarantee of future returns.*

#### ML v3 (auto-retrains on startup, MODEL_VERSION="v3")
- Added 3 external macro features to RF skip-day filter (16 → 19 features):
  - `prev_vix_level` — lag r=-0.130***, VIX≥30 yesterday → +0.267% USD/ILS today
  - `prev_dxy_return` — lag r=-0.126***
  - `prev_10y_yield` — same-day r=+0.305*** (market context)
- New helper `MLSkipFilter._fetch_external_indicators()` → fetches `^VIX`, `DX-Y.NYB`, `^TNX`
- Wired into both `train()` and `predict_today()` pipelines
- Backward compatible: neutral fill values if external data unavailable
- **Walk-forward backtest (13 folds, 2023–2026):**
  | | Baseline | v2 (16 feat) | v3 (19 feat) |
  |--|--|--|--|
  | Win Rate | 45.1% | 77.4% | 77.2% |
  | Profit Factor | 0.82 | 3.42 | 3.39 |
  | Trades | 832 | 327 | 329 |
  - Δv3-v2: PF -0.03, WR -0.2% → **statistically tied (noise)**
  - 3 new features rank above all calendar features in importance (4.7% combined)
  - Verdict: keep v3 — not better today, but right model for the future (VIX bias visible in extremes)

#### Chart interactivity (candle chart)
- Crosshair: vertical + horizontal hairlines with live price label (Y-axis) and time label (X-axis)
- Mouse wheel zoom anchored to cursor position
- Click-and-drag pan across candle history
- `1:1` reset zoom button appears when zoomed/panned
- TF change auto-resets zoom

#### Admin controls
- `ADMIN` yellow badge in header
- `👁 User View` toggle button — admins can preview the dashboard as a regular user
- Body classes drive visibility:
  - `body.is-admin` — reveals all `.admin-only` elements
  - `body.user-preview-mode` — hides them, shows yellow banner
- `.admin-only` elements:
  - SMART ▼ data source selector (FXCM/12Data/FCS/BOI architecture)
  - Layout + Reset Baseline buttons
  - `lastUpdate` source indicator
  - KB: Data Sources section (file names, API endpoints, limits)
  - KB: AI/ML Algorithms section (model details, features, training paths)
  - About/Version modal
  - Admin Panel button (links to `/admin`)

#### Infrastructure
- `auth.py`: `FIREBASE_SERVICE_ACCOUNT_JSON` env var fallback for Cloud Run (file → env → bypass)
- `firebase-service-account.json` added to `.gitignore` (security fix)
- Firebase: `127.0.0.1` added to authorized OAuth domains (enables Google sign-in on local dev)
- GitHub secret `FIREBASE_SA_JSON` set (fixes admin panel "No users registered yet")

#### Responsive layout
- <1350px: hide secondary header buttons (Layout, Reset Baseline, Simulate, lastUpdate, hdr-badge)
- <1200px: compress grid columns + reduce padding
- <960px: 2-column grid (price/bounds | gauge | candles | news/signals)
- <640px: mobile single column stack
- Header badge hides at <1350px to save space

#### Docs
- `CHANGELOG.md` — full v1.8.0 + v1.8.1 entries + backfill v1.7.4, v1.0.0
- `my_guide.md` — new "Signals & Market Correlations" section with 5yr empirical stats tables
- Release-notes policy header: every change → both CHANGELOG.md AND my_guide.md

---

## Lessons Learned (must read before next session)

### 1. **Worktree partial-copy trap** ⚠️ CRITICAL
- The `.claude/worktrees/musing-haibt/` was a *partial* copy of the repo (3,251 lines of `index.html` vs 4,859 on main)
- Copying files from worktree to main wiped ~1,600 lines of UI (Tour, Simulate, AI mini panel, candle TF buttons, data source manager, etc.)
- **Rule going forward:**
  - Before any `cp` from worktree → main, run `wc -l` on both versions
  - Prefer `sed -i` for targeted text replacements, never full file overwrites
  - Or: cherry-pick commits directly onto a clean `git checkout -b feature/X origin/main`

### 2. **auth.py imports must mirror app.py exactly**
- A "clean refactor" of `auth.py` dropped `firestore` import and `get_firestore()` function
- `app.py` imports both → `ImportError` → 503 Service Unavailable in production
- **Rule:** Before merging any auth.py change, `grep "from auth import" app.py` and verify every name still exists

### 3. **Version strings are hardcoded in 4 places — update all together**
- `APP_VERSION` const in `templates/index.html` (line ~2245)
- About modal badge text: `>v1.7.4<` in 2 places
- `apps/fx/package.json` → `"version": "1.7.4"`
- New entry at top of `APP_CHANGELOG` array
- **Rule:** Bump them in a single commit titled `chore: bump version to vX.Y.Z`

### 4. **Cloud Run secrets ≠ GitHub secrets**
- The deploy workflow writes `FIREBASE_SA_JSON` (GitHub secret) → `firebase-service-account.json` on container
- If GitHub secret is missing/empty, deploy succeeds but app runs in BYPASS mode (no users visible)
- Always check: `gh secret list --repo avirammeidan-hash/FX-Range-Master`

### 5. **Firebase OAuth needs domain whitelist**
- Local dev on `127.0.0.1:5000` requires adding `127.0.0.1` to Firebase Console → Authentication → Settings → Authorized domains
- `localhost` is whitelisted by default; `127.0.0.1` is NOT

### 6. **PR diffs lie when source branch is partial**
- `git diff origin/main..HEAD` showed 23,981 deletions when we only intended ~500
- Always check `git diff --stat` before opening a PR
- If unexpected files appear in the diff → DO NOT MERGE → cherry-pick onto a fresh branch instead

### 7. **CI doesn't run on PRs (only on push to main)**
- `.github/workflows/deploy.yml` triggers only on `push: branches: [main]`
- PRs show no checks — you only learn about failures *after* merging
- **Mitigation:** Local sanity check before merge: `python -c "import app, auth, ml_filter"` to catch ImportErrors

### 8. **CHANGELOG + KB are the source of truth**
- User-defined policy: every code change goes in BOTH `CHANGELOG.md` AND `my_guide.md`
- Skip this and the dashboard's About modal goes stale (it reads from `APP_CHANGELOG` JS array)

### 9. **Performance modal vs KB**
- `openHelp()` → Performance modal (read-only KPIs, NO ML details)
- `openKB()` → Knowledge Base (data sources, ML algorithms — admin-only sections)
- They look similar but are different modals — verify which one you're editing

### 10. **Cloud-modification safety (user preference, global rule)**
- Never modify cloud state (ADX, Cloud Run, GitHub repos, Firestore, etc.) without **asking twice**:
  1. State the action: "I'm about to X. Confirm?"
  2. Wait for explicit yes
  3. Ask again: "Last chance — proceeding with X?"
  4. Only proceed on second yes
- Applies to: `gh secret set`, `gh pr merge`, `gh workflow run`, Firebase Console changes, ADX commands, `az` mutations

---

## Parked work / Future enhancements

### High value (next session candidates)

1. **VIX as a regime filter (not just ML feature)**
   - v3 walk-forward showed VIX features are tied with v2 in normal regimes
   - But the empirical edge is in **extremes**: VIX≥30 → +0.267% bias, VIX≥25 → 1.86% std dev (3× normal)
   - Implementation: if `vix_level >= 25`: widen `half_width_pct *= 1.5` or skip entirely
   - Estimated impact: large in crisis weeks (2022 Ukraine, 2023 banking, etc.)

2. **Feature Importance chart in KB**
   - Currently hardcoded (`abs_gap_pct 40.1%`, `gap_pct 15.3%`, etc.)
   - Should fetch live from `/api/ml-status` or similar
   - Will then show v3's actual `prev_vix_level: 1.6%`, `prev_dxy_return: 1.58%`, `prev_10y_yield: 1.56%`

3. **CFTC COT for ILS futures**
   - Mentioned in ULTRON_COMPARE — weekly, free, downloadable
   - Shows institutional long/short positioning on ILS
   - Unique leading indicator no one else uses
   - Source: https://www.cftc.gov/dea/futures/financial_lf.htm

4. **Fear & Greed Index overlay**
   - Free API: https://api.alternative.me/fng/
   - 0–100 composite; extreme fear historically volatile for ILS
   - Shows in dashboard as warning badge (no ML wiring needed initially)

5. **HMM regime detection on USD/ILS itself**
   - ULTRON uses HMM on SPY volatility
   - For us: detect range-bound vs trending regimes on USD/ILS
   - Critical because mean-reversion strategy only works in range regimes
   - Library: `hmmlearn` (Python)

6. **Trump / WH RSS monitor**
   - Tariff/sanctions statements → immediate USD/ILS impact
   - ULTRON polls Truth Social every 15s
   - For us: lighter — RSS check every 5 min

### Lower priority

7. **FinBERT sentiment** (replace keyword matching in `news_monitor.py`)
8. **BoI Intervention Flag** — Bank of Israel buys USD → overrides all signals
9. **FEDFUNDS rate change calendar** — Fed hike/cut days have outsized USD/ILS moves
10. **Update Performance modal KPI cards dynamically** — currently hardcoded (73%, PF=2.55)

---

## Useful commands (for next session)

```powershell
# Local dev start
cd C:\workgit\FX-Range-Master\.claude\worktrees\musing-haibt
python app.py
# → http://127.0.0.1:5000

# Verify auth.py imports match app.py
grep "from auth import" app.py
grep "^def " auth.py

# Cherry-pick session commits onto clean PR branch
git checkout -b feature/X origin/main
git cherry-pick <commit1> <commit2> ...
git diff origin/main --stat   # sanity-check before push

# Watch deploy
gh run watch $(gh run list --repo avirammeidan-hash/FX-Range-Master --limit 1 --json databaseId -q '.[0].databaseId') --repo avirammeidan-hash/FX-Range-Master --exit-status

# Check live logs
gcloud run services logs read fx-range-master --region me-west1 --project fx-range-master --limit 50

# ML retrain test locally
python -c "from ml_filter import MLSkipFilter; m=MLSkipFilter(); print(m.train(retrain=True))"

# Pre-deploy smoke test (run before every push)
powershell -ExecutionPolicy Bypass -File scripts\local_check.ps1

# Pytest baseline
python -m pytest tests/ -v
```

---

## Production incident response

### Rollback to previous Cloud Run revision (30-second recovery)

Cloud Run keeps every deployed revision. If a deploy breaks production:

```bash
# 1. List recent revisions
gcloud run revisions list \
  --service fx-range-master --region me-west1 --project fx-range-master --limit 5

# 2. Roll traffic back to the previous good revision
gcloud run services update-traffic fx-range-master \
  --region me-west1 --project fx-range-master \
  --to-revisions PREVIOUS_REVISION_NAME=100

# Or for instant rollback to revision N-1 by name pattern:
gcloud run services update-traffic fx-range-master \
  --region me-west1 --project fx-range-master \
  --to-latest=false --to-revisions=fx-range-master-<HASH>=100
```

### Debugging a failed deploy
```bash
# Last 50 log lines (catches ImportError, Firebase init failures, etc.)
gcloud run services logs read fx-range-master \
  --region me-west1 --project fx-range-master --limit 50
```

### After rollback: ship the forward fix
1. Local: reproduce the error (`python -c "import app"`)
2. Local: fix + `pytest tests/` until green
3. Local: `powershell -File scripts\local_check.ps1`
4. PR → CI green → merge → auto-deploy
5. Then point traffic back to latest (`--to-latest=true`)

---

## Repo facts (for context)

- **Repo:** `avirammeidan-hash/FX-Range-Master` on GitHub
- **Deploy:** Cloud Run `fx-range-master` in `me-west1`, auto-deploys on push to main
- **Git user:** `avirammeidan-hash` (HTTPS auth, PAT in keyring)
- **Firebase project:** `fx-range-master`
- **Admin emails (config.yaml → firebase.admin_emails):**
  - `aviram.meidan@gmail.com`
  - `avirammeidan@gmail.com`
- **Worktree:** `C:\workgit\FX-Range-Master\.claude\worktrees\musing-haibt` (⚠️ partial — see Lesson #1)
  - **Branch:** `claude/musing-haibt` (stale, never merged — work was cherry-picked then we kept shipping forward fixes)
  - **To remove safely** when no longer needed:
    ```bash
    cd C:\workgit\FX-Range-Master
    # 1. Verify no uncommitted work you want to keep
    cd .claude/worktrees/musing-haibt && git status && cd ../../..
    # 2. Remove the worktree
    git worktree remove .claude/worktrees/musing-haibt --force
    # 3. Delete the stale branch
    git branch -D claude/musing-haibt
    ```
- **Main repo:** `C:\workgit\FX-Range-Master`
- **Related projects:**
  - `C:\workgit\ULTRON_COMPARE` — US equity bot, 96 AI modules, FRED/VIX/intermarket — good source of ideas (see "Parked work")
  - `C:\workgit\Stocks` — frontend playground
  - `C:\workgit\_Capabilities` — user's MCP/tools config
