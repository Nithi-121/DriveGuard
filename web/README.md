# DriveGuard on Vercel

This is the static, browser-side edition of the DriveGuard dashboard. The original Streamlit app remains in `../app/streamlit_app.py` for local and Streamlit-hosted use. This directory is the Vercel project root.

## Deploy from the GitHub repository

1. Import `Nithi-121/DriveGuard` into Vercel.
2. Set **Root Directory** to `web`.
3. Set **Framework Preset** to `Other`.
4. Leave **Build Command** blank. Set **Output Directory** to `public` (also declared in `vercel.json`).
5. Deploy.

No server, API secrets, or environment variables are required. The dashboard is static and loads `public/data/driveguard.json` in the visitor's browser.

## Refresh its published data

From the repository root, regenerate the compact browser data package from the checked-in complete test predictions and report outputs:

```powershell
.\.venv\Scripts\python.exe web\build_data.py
```

Commit and push changes to `web/public/`; the connected Vercel project will build a new deployment. The JSON keeps the full-test model metrics and daily aggregates, plus a clearly labeled risk review subset and sampled drive histories. The review subset is not used to recalculate test performance.

## Verify locally

From the repository root, start a static HTTP server:

```powershell
.\.venv\Scripts\python.exe -m http.server 8765 --directory web/public
```

Open `http://localhost:8765/`. The app uses no external JavaScript, charting, font, or icon services.
