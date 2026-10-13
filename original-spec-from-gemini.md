Here is a complete, structured project specification and prompt that you can copy and paste directly into Claude:

```markdown
# Autonomous Multi-Asset AI Paper Trading System (Project Specification)

## 1. Project Objective & Philosophy
Build an autonomous, modular 24/7 AI-driven trading system in Python. 

### Core Operating Principles:
1. **Safety & Paper Trading First:** The system must start with 100% virtual paper trading using a simulated bankroll of **€50.00**, but connected to **live real-time market data**.
2. **Realistic Expectations:** There are no "guaranteed profits" in financial markets. The primary goal is capital preservation, statistical risk management, and rigorous testing before ever using real capital.
3. **Execution Separation:** Revolut is NOT used for automated trading due to lack of public retail APIs and ban risks. Architecture must be modular so that execution routes either to a local paper ledger or official broker APIs (such as Alpaca or CCXT/Binance testnet).

---

## 2. Asset Classes & Time Horizons (Specialist Council)
The system should accommodate multiple asset classes through modular specialist agents rather than a single monolithic prompt:
* **Crypto (24/7):** High volatility, short-term momentum, scalping, and breakout trades (e.g., BTC, ETH, SOL).
* **Stocks / Equities (Market Hours):** Swing trading (days to weeks), corporate earnings, and fundamentals (e.g., AAPL, NVDA, SPY).
* **Forex (24/5):** Macroeconomic trends and currency pairs (e.g., EUR/USD).

---

## 3. System Architecture & Modules

### A. Live Market Data Ingestion (`data_feed.py`)
* **Crypto:** Free public live WebSocket / REST data via `ccxt` (Binance, Coinbase, Kraken). No API keys required for public order books/candles.
* **Stocks & Forex:** Real-time / intraday feeds via `yfinance` or Alpaca Market Data.
* **Technical Indicators:** Compute standard quantitative signals (RSI, EMA 20/50/200, MACD, Volume Spikes, ATR).

### B. Multi-Model Consensus & Debate Engine (`ensemble_agent.py`)
To prevent hallucinations and false signals, decisions are evaluated by an ensemble of low-latency, cost-effective LLMs (e.g., Google Gemini 2.0 Flash, OpenAI GPT-4o-mini, Anthropic Claude 3.5 Haiku):
1. **Consensus Voting:** A trade is only executed if at least 2 out of 3 models agree on the action (`BUY`, `SELL`, or `HOLD`).
2. **Bull vs. Bear Debate:** One model identifies upside arguments; another model searches for pitfalls, false breakouts, and liquidity traps.
3. **Structured Outputs:** All models must return strict JSON schemas:
   ```json
   {
     "ticker": "BTC/USDT",
     "action": "BUY | SELL | HOLD",
     "confidence": 0.85,
     "thesis": "RSI oversold divergence on 1H with above-average buy volume",
     "suggested_stop_loss_pct": 2.0,
     "suggested_take_profit_pct": 4.0
   }
   ```

### C. Self-Reflection & Mistake-Learning Loop (`memory_loop.py`)
LLMs do not retain memory across runs by default. This module ensures the AI learns from losing trades:
1. **Post-Mortem Analysis:** When a paper trade exits (hits stop-loss or take-profit), the system performs an automated review comparing the entry thesis with what actually occurred.
2. **Epistemic Memory Store:** Mistakes and lessons learned are recorded in a local database / JSON file (e.g., *"Bought breakout during low weekend volume; failed to confirm support"*).
3. **Prompt Injection:** Before analyzing a new trade, the top 3 most relevant recent mistakes are injected into the agent's prompt to avoid repeating errors.

### D. Deterministic Hardcoded Risk Controls (`risk_manager.py`)
The LLM is NEVER allowed to decide risk parameters autonomously. The Python code strictly enforces:
* **Max Position Size:** Maximum 10%–20% of total portfolio balance per single trade (e.g., €5 to €10 per trade on a €50 balance).
* **Hard Stop-Loss:** Automatic exit at -2.0% loss. Cannot be overridden.
* **Take-Profit:** Automatic exit at +4.0% profit (2:1 reward-to-risk ratio).
* **Daily Circuit Breaker:** If the account experiences a cumulative loss of 5% in a rolling 24-hour window, all trading pauses for 24 hours.

### E. Multi-Asset Virtual Paper Ledger (`portfolio.py`)
* Initialized with **€50.00 virtual cash**.
* Simulates real-world execution, accounting for estimated trading fees (e.g., 0.1%) and slippage.
* Logs complete trade history and calculates performance metrics: Win Rate, Profit/Loss, Sharpe Ratio, and Max Drawdown.

---

## 4. Deployment Strategy
* **Phase 1 (Development & Local Testing):** Run and debug locally in Python.
* **Phase 2 (24/7 Cloud Execution):** Deploy to an always-on Cloud VPS (e.g., Oracle Cloud Always-Free Tier, DigitalOcean, or Hetzner ~€4/mo) to keep the bot monitoring markets 24/7/365.

---

## 5. Implementation Roadmap
Please implement this project modularly in Python with the following file structure:
1. `config.py` — Settings, starting balance (€50), risk limits, and API keys.
2. `data_feed.py` — Live data fetcher (crypto via ccxt, stocks/forex via yfinance/Alpaca) and technical indicator calculations.
3. `portfolio.py` — SQLite/JSON virtual paper trading ledger tracking balances, positions, and history.
4. `risk_manager.py` — Deterministic order validation, stop-loss/take-profit monitors, and daily circuit breaker.
5. `ensemble_agent.py` — Multi-model prompt orchestration (Gemini, Claude, GPT) and consensus voting.
6. `memory_loop.py` — Post-mortem trade analysis and persistent lesson store.
7. `main.py` — Asynchronous main event loop running 24/7.
```