@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo .venv is missing. Run setup.bat first.
  exit /b 1
)

".venv\Scripts\python.exe" "swing_screener_6_patterns.py" ^
  --min-score 52 ^
  --max-risk-pct 5 ^
  --min-reward-risk 1.0 ^
  --min-rel-volume 0.30 ^
  --min-hourly-dollar-volume 75000 ^
  --min-price 1 ^
  --pullback-touch-pct 4 ^
  --pullback-max-ema20-distance-pct 7 ^
  --pullback-volume-multiplier 1.50 ^
  --breakout-event-volume-multiplier 0.70 ^
  --breakout-retest-tolerance-pct 4 ^
  --breakout-max-extension-pct 6 ^
  --range-support-distance-pct 5 ^
  --reversal-higher-low-pct 0.10 ^
  --reversal-structure-break-pct 0 ^
  --reversal-min-rel-volume 0.45 ^
  --allow-resistance-before-target ^
  --allow-neutral-candle ^
  --allow-reversal-below-ema50 ^
  --allow-uptrend-continuation ^
  --debug

endlocal
