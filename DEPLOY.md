# Deploy this on your phone with Streamlit Community Cloud

## What you need

- A GitHub account
- A Streamlit Community Cloud account
- This project folder uploaded to a **private** GitHub repository

The app uses no API key and stores no brokerage credentials. It retrieves chart data from Yahoo through `yfinance` only when you tap **Run scan**.

## 1. Create a private GitHub repository

1. Go to GitHub and sign in.
2. Click the `+` menu in the upper-right, then **New repository**.
3. Repository name: `swing-screener`
4. Set it to **Private**.
5. Do not add a README or `.gitignore` because this package already includes them.
6. Click **Create repository**.

## 2. Upload this project

1. Extract the ZIP.
2. Open the new GitHub repository.
3. Click **Add file** → **Upload files**.
4. Drag in the contents of the extracted project folder:
   - `app.py`
   - `swing_screener_6_patterns.py`
   - `watchlist.txt`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `.gitignore`
5. Do **not** upload `.venv` or any local `output` folder.
6. Click **Commit changes**.

## 3. Deploy

1. Open Streamlit Community Cloud and sign in with GitHub.
2. In the workspace, click **Create app**.
3. Select:
   - Repository: your private `swing-screener` repository
   - Branch: `main`
   - Main file path: `app.py`
4. Choose an app URL such as `hassan-swing-screener`.
5. Click **Deploy**.

## 4. Make it private

Because the source repository is private, the app starts private to workspace developers by default.

Do not change it to public. If you later want someone else to access it, use the Streamlit **Share** control to invite them deliberately.

## 5. Use it from your phone

1. Open the resulting `https://<your-name>.streamlit.app` URL on your phone.
2. Sign in if Streamlit asks.
3. Save the page to your home screen through the browser's **Add to Home Screen** action.
4. Open it, adjust profile/watchlist if needed, and tap **Run scan**.

## Watchlist edits

- Editing the Watchlist box in the app affects the current browser session.
- To make a permanent default change, edit `watchlist.txt` in GitHub and commit it.
- Streamlit Community Cloud redeploys changes after you commit to the repository.

## Local test before deploying

On Windows:

```powershell
.\setup_local.bat
.un_local.bat
```

Then open the shown local URL in your browser.

## Private repository access note

When connecting a private GitHub repository, Streamlit may ask for GitHub permissions to access private repositories. Review the authorization request before accepting it.
