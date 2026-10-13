# Running it on Render (free)

You do these parts, because they need your accounts. Nothing here costs money. Don't add a card anywhere.

## 1. Put the code on GitHub (private)
1. github.com > New repository > name `ultron-trader`, **Private**, no README. Create it.
2. In the project folder, in a terminal:
   ```
   git remote add origin https://github.com/<you>/ultron-trader.git
   git push -u origin main
   ```
   (`.env` and the database are git-ignored, so your keys do not go up.)
3. A second private repo for backups, e.g. `ultron-state` (tick "Add a README" so it has a branch).
4. github.com > Settings > Developer settings > Fine-grained tokens > new token. Repository access: only
   `ultron-state`. Permission: Contents, read and write. Copy the token.

## 2. Render
1. render.com > sign up with GitHub (free plan, no card).
2. New > **Web Service** (not Blueprint) > "Build and deploy from a Git repository" > connect GitHub, allow access
   to `ultron-trader` > select it.
3. Settings: Language **Python 3**, Branch `main`, Instance Type **Free**.
   Build Command: `pip install -r requirements.txt`   Start Command: `python main.py`
   Health Check Path (under Advanced): `/health`
4. Environment variables (Advanced > Add):
   - `PYTHON_VERSION` = `3.12.7`
   - `GEMINI_API_KEY`, `RESEND_API_KEY`, `EMAIL_TO` (same values as your `.env`)
   - `GEMINI_MODEL` = `gemini-2.5-flash`, `XAI_DAILY_CALLS` = `0`
   - `DASHBOARD_TOKEN`: any long random string. The dashboard refuses to start in the cloud without one.
   - `BACKUP_REPO` = `youssef-110/ultron-state`, `BACKUP_GITHUB_TOKEN` = the token from step 1.4
   - optional: `CEREBRAS_API_KEY` (starts with `csk-`)
5. Create Web Service. The first build takes a few minutes. Logs show "Ultron online".

## 3. Keep it awake and get told when it dies
UptimeRobot (free) > Add monitor > HTTP(s) > `https://<app>.onrender.com/health` > every 5 minutes, alert to your email.
Render's free service sleeps after 15 minutes without a request. This pings it. `/health` also turns red if the
trading loop has been silent for 20 minutes, so you get an email.

## 4. Look at it
`https://<app>.onrender.com/?key=<DASHBOARD_TOKEN>` and `/ultron?key=<DASHBOARD_TOKEN>`.
Press "Send a test email" on the dashboard to check alerts from the cloud copy.

## Things to know
- Free instances get about 750 hours a month, so one always-on service fits, two don't.
- On every restart the bot pulls the last backup (ledger, lessons, scores) before it starts, and carries on.
  Backups happen after each debate or trade, and hourly otherwise.
- **Stop the bot on your PC before the cloud one starts**, or two copies will trade the same pretend account
  and send double emails. The cloud copy starts fresh (50 paper euros) unless the backup repo already holds a ledger, so
  nothing from your PC's database carries over.
- Check the model names in the Render env against Google AI Studio now and then. They change.
