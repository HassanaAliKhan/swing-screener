# Hourly Swing Screener — phone web app

This is a Streamlit front-end for the six-pattern hourly swing scanner.

## What the app does

- Opens in any browser, including phone
- Accepts an editable watchlist
- Runs the scan only when you tap **Run scan**
- Shows only potential long-entry candidates
- Shows Entry, Stop, 5% Target, Risk %, Reward/Risk, relative volume, and resistance distance
- Allows direct CSV downloads
- Optionally shows all classifications and Yahoo data errors

## Deployment

Read [`DEPLOY.md`](DEPLOY.md). The intended hosting path is:

```text
Private GitHub repository → private Streamlit Community Cloud app → phone bookmark
```

## Local Windows test

```powershell
.\setup_local.bat
.un_local.bat
```

## Important limits

- The app uses Yahoo/yfinance data. It can be delayed, unavailable, or occasionally malformed.
- The last provider candle is ignored because it may be in progress.
- The result table is a screening queue, not an automatic buy list.
- Verify live price, daily resistance, earnings, news, and market context before acting.
