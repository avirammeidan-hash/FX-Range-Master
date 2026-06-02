# local_check.ps1 -- Pre-deploy smoke test
# Run this BEFORE every git push to catch issues that would crash production.
# Usage: powershell -File scripts\local_check.ps1

$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "================ Pre-deploy smoke check ================" -ForegroundColor Cyan

# 1. Python imports -- catches ImportError that would 503 in prod
Write-Host ""
Write-Host "[1/4] Python imports..." -ForegroundColor Yellow
$pyResult = python -c "import app, auth, ml_filter, scanner, news_monitor, events; print('imports OK')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ("  FAIL: " + $pyResult) -ForegroundColor Red
    exit 1
}
Write-Host "  OK" -ForegroundColor Green

# 2. Syntax check on all main Python files (matches CI lint job)
Write-Host ""
Write-Host "[2/4] Python syntax (py_compile)..." -ForegroundColor Yellow
$pyFiles = @('app.py', 'auth.py', 'ml_filter.py', 'scanner.py', 'news_monitor.py', 'events.py', 'engine.py')
foreach ($f in $pyFiles) {
    if (Test-Path $f) {
        python -m py_compile $f 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host ("  FAIL: " + $f + " has syntax error") -ForegroundColor Red
            exit 1
        }
    }
}
Write-Host ("  OK -- " + $pyFiles.Count + " files checked") -ForegroundColor Green

# 3. Template sanity -- verify index.html has expected anchors
Write-Host ""
Write-Host "[3/4] Template sanity..." -ForegroundColor Yellow
$tpl = Get-Content 'templates/index.html' -Raw
$required = @('USD/ILS RANGE INTELLIGENCE', 'Educational tool', 'candleChart', 'Firebase SDK', 'ADMIN_EMAILS')
foreach ($needle in $required) {
    if (-not $tpl.Contains($needle)) {
        Write-Host ("  FAIL: '" + $needle + "' missing from index.html") -ForegroundColor Red
        exit 1
    }
}
Write-Host ("  OK -- " + $required.Count + " anchors present") -ForegroundColor Green

# 4. App boot test (5 sec)
Write-Host ""
Write-Host "[4/4] App boot test (5 sec)..." -ForegroundColor Yellow
$proc = Start-Process -FilePath 'python' -ArgumentList 'app.py' -PassThru -WindowStyle Hidden -RedirectStandardOutput 'local_check_app.log' -RedirectStandardError 'local_check_app.err'
Start-Sleep -Seconds 5
$alive = -not $proc.HasExited
try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
Remove-Item 'local_check_app.log','local_check_app.err' -ErrorAction SilentlyContinue

if (-not $alive) {
    Write-Host "  FAIL: app.py exited immediately -- check logs" -ForegroundColor Red
    exit 1
}
Write-Host "  OK -- started cleanly" -ForegroundColor Green

Write-Host ""
Write-Host "================ All checks passed ================" -ForegroundColor Green
Write-Host "Safe to commit and push." -ForegroundColor Green
Write-Host ""
