# Ultron: a paper-trading desk run by AI agents

A 24/7 bot that watches crypto and US stocks on live data, lets a small team of AI agents argue about every setup, and trades a fake €500 account. No real money moves anywhere. It costs €0 to run if you stick to the free tiers.

## Who's on the desk

| Agent | What it does | Uses an LLM? |
|---|---|---|
| **Scout** | Scans every closed 1h candle and flags setups (oversold dip in an uptrend, EMA20 reclaim on volume, MACD cross above EMA50) | No |
| **Analyst** | Crypto or stock specialist, gives the first read | Gemini |
| **Bull** | Makes the case for the trade | Grok (falls back to Gemini) |
| **Bear** | Reads the bull's argument and tears into it | Gemini |
| **Ultron** | The manager. Only gets asked if at least 2 of the 3 lean BUY. Sees every agent's track record, the desk's history with this exact setup, and past lessons, then makes the call | Gemini |
| **Historian** | Writes a post-mortem lesson after every closed trade | Gemini |
| **Risk** | Sizes the trade, sets the stop and target, can veto anything. Ultron can't override it | No |

The LLMs never touch risk. Every trade is 20% of equity, exits at -2% or +4%, max 3 open positions, no re-entry for 6 hours after a stop, and a 5% drawdown in 24h shuts new entries down for a day. All of that is plain Python in `risk_manager.py` and `config.py`.

## How it learns

The models don't retrain themselves, so learning here means three concrete things, all kept in SQLite:

1. **Lessons.** The Historian writes one after every trade. Before each decision, the three most relevant ones (same setup, same market regime, same coin) go into the prompts.
2. **Shadow tracking.** Every setup the Scout flags gets followed for 48h to see if it would've hit +4% or -2% first, including the ones Ultron passed on. That's way more feedback than the handful of real trades, and it shows missed winners, not only losses.
3. **Scoreboard.** Shadow outcomes grade every agent's vote. An agent that keeps calling losers gets less weight in the council.

`report.json` has the number that matters most: win rate when Ultron said BUY versus win rate of every Scout setup. If Ultron isn't beating the baseline after a few weeks, the AI isn't adding anything yet, and that's worth knowing. Break-even is around 38% with these exits and fees.

## Staying at €0

- **Gemini**: free tier, rate-limited. Keep billing off on the Google project and it can't charge you. Note that on the free tier Google may use your prompts to improve its models. That's fine here since it's only market data.
- **Grok**: only the promo credits are free, and they expire. Don't add a card in the xAI console. When the credits run out the API starts refusing, the bot switches Grok off for the day and Gemini plays the bull.
- Every call is counted in the database before it's sent, with a hard daily cap per provider (200 Gemini, 40 Grok by default). It survives restarts. When the cap is hit the desk just holds.
- The LLMs only wake up when the Scout finds something, so a normal day is a few dozen calls, not thousands.

## Run it

On Windows the easy way is to double-click `setup.bat`. It copies the project into `Projects\ultron-trader`, sets up Python, runs the tests, makes the first git commit and opens VS Code. Or do it by hand:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste your keys in
python -m pytest -q tests          # should say 24 passed
python main.py
```

It wakes up every 5 minutes, but only thinks when a new hourly candle closes. Watch `trader.log` for decisions and `report.json` for the scoreboard. Stopping and restarting is safe; everything lives in `paper_ledger.db`. Delete that file to start over with a fresh €500.

Check the model names in `.env` against what your Google AI Studio and xAI console actually list. Both companies rename models often.

## Known limits

- Long only. A SELL from the models only closes a position; there's no shorting.
- Forex isn't in yet. Trading EUR/USD from a EUR account needs separate P&L handling.
- yfinance is unofficial and its intraday data can lag. Fine for paper trading, not for real money.
- US market hours are hardcoded and don't know about holidays.
- A few weeks of results prove nothing. You want a few hundred shadow outcomes before trusting any win rate.

## Files

```
config.py        every setting and risk number
data_feed.py     Kraken (ccxt) + yfinance, converted to EUR, plus indicators
portfolio.py     SQLite paper ledger: cash, positions, trades, metrics
risk_manager.py  sizing, hard stops, circuit breaker, cooldown
llm.py           Gemini + Grok clients, daily budget, fallback routing
agents.py        Scout, Analyst, Bull, Bear, Ultron, Historian
memory_loop.py   lessons, shadow tracking, agent scoreboard
main.py          the 24/7 loop
tests/           24 tests, no network or keys needed
```
