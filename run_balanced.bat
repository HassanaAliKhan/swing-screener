@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv is missing. Run setup.bat first.
  exit /b 1
)

".venv\Scripts\python.exe" "swing_screener_6_patterns.py" ^
  --min-score 58 ^
  --max-risk-pct 4.5 ^
  --min-reward-risk 1.1 ^
  --min-rel-volume 0.45 ^
  --min-hourly-dollar-volume 100000 ^
  --min-price 1 ^
  --pullback-touch-pct 3 ^
  --pullback-max-ema20-distance-pct 5 ^
  --pullback-volume-multiplier 1.30 ^
  --breakout-event-volume-multiplier 0.90 ^
  --breakout-retest-tolerance-pct 3 ^
  --breakout-max-extension-pct 4 ^
  --range-support-distance-pct 4 ^
  --reversal-higher-low-pct 0.25 ^
  --reversal-structure-break-pct 0 ^
  --reversal-min-rel-volume 0.60 ^
  --allow-resistance-before-target ^
  --debug

endlocal
