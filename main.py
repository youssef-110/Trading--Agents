"""24/7 loop. Run with:  python main.py

Each cycle (every 5 minutes by default):
  1. Fetch prices for the universe (crypto always, US stocks during market hours).
  2. Risk enforces hard stop-loss / take-profit on open positions.
  3. The Historian writes a lesson for every trade that just closed.
  4. Circuit breaker check (halts new entries after a 5% drop from the 24h high).
  5. Shadow setups get graded, which updates every agent's track record.
  6. On each newly CLOSED 1h candle: Scout scans; if it flags a setup, Ultron runs
     the council; Risk approves or vetoes; the paper ledger executes.
  7. report.json is rewritten so you can see what's going on.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from agents import COUNCIL, DEBATE_CALLS, Historian, Scout, Ultron
from config import SETTINGS, Settings
from llm import CallBudget, LLMRouter
from backup import Backup
from notify import Notifier
from memory_loop import ActivityLog, LessonStore, Scoreboard, ShadowTracker
from portfolio import Portfolio, utcnow
from risk_manager import Bar, RiskManager, Signal

log = logging.getLogger("main")
NEW_YORK = ZoneInfo("America/New_York")


def us_market_open(now: datetime) -> bool:
    """Regular session only. Doesn't know about US holidays; yfinance just won't move then."""
    et = now.astimezone(NEW_YORK)
    return et.weekday() < 5 and time(9, 30) <= et.time() < time(16, 0)


def bar_since(df: pd.DataFrame, opened_at: str) -> Bar:
    """Low/high of the bars that started after the position opened, plus the latest price."""
    recent = df[df.index >= pd.Timestamp(opened_at)]
    last = float(df["close"].iloc[-1])
    return Bar(last=last, low=min([last, *recent["low"].tolist()]),
               high=max([last, *recent["high"].tolist()]))


class Bot:
    def __init__(self, settings: Settings, data, providers: dict, limits: dict,
                 notifier: Notifier | None = None, backup: Backup | None = None):
        self.s = settings
        self.data = data
        self.notifier = notifier or Notifier(None)
        self.backup = backup or Backup(None, settings.db_path)
        self.scanned = 0                       # new candles the Scout looked at this cycle
        self.debates = 0                       # debates started this cycle
        self.pf = Portfolio(settings.db_path, settings.starting_balance_eur, settings.costs)
        self.risk = RiskManager(self.pf, settings.risk, settings.practice)
        db = self.pf.db
        self.router = LLMRouter(providers, CallBudget(db), limits)
        self.lessons = LessonStore(db)
        self.scores = Scoreboard(db)
        self.shadows = ShadowTracker(db, self.scores)
        self.activity = ActivityLog(db)
        self.notifier.on_result = lambda subject, ok, detail: self.activity.email(utcnow(), subject, ok, detail)
        self.scout = Scout(settings.scout)
        self.ultron = Ultron(self.router, self.lessons, self.scores, self.shadows)
        self.historian = Historian(self.router, self.lessons)

    async def _fetch(self, now: datetime, stocks_open: bool) -> tuple[dict, list]:
        held = {p.symbol for p in self.pf.open_positions()}
        universe = [(s, "crypto") for s in self.s.crypto_symbols]
        universe += [(s, "stock") for s in self.s.stock_symbols if stocks_open or s in held]
        frames, failed = {}, []
        for sym, ac in universe:
            try:
                frames[sym] = (ac, await self.data.bars_eur(sym, ac))
            except Exception as e:  # one bad feed shouldn't stop the desk
                log.warning("data for %s failed: %s", sym, e)
                failed.append(sym)
        return frames, failed

    async def cycle(self, now: datetime | None = None) -> None:
        now = now or utcnow()
        stocks_open = us_market_open(now)
        self.scanned = self.debates = 0
        self.beat()
        self.pf.set_state("live_status", "fetching prices")
        frames, failed = await self._fetch(now, stocks_open)
        prices = {s: float(df["close"].iloc[-1]) for s, (_, df) in frames.items()}

        # Hard exits first, always.
        bars = {p.symbol: bar_since(frames[p.symbol][1], p.opened_at)
                for p in self.pf.open_positions() if p.symbol in frames}
        for t in self.risk.check_exits(bars, now):
            log.info("CLOSED %s %s %+.2f EUR (%+.2f%%)", t.symbol, t.exit_reason, t.pnl_eur, t.return_pct)
            await self.notifier.trade_closed(
                t.symbol, t.pnl_eur, t.return_pct, t.exit_reason, t.entry_price, t.exit_price,
                self.pf.equity({k: float(f["close"].iloc[-1]) for k, (_, f) in frames.items()}))
            await self.historian.post_mortem(t, now)
            await self.backup.maybe_backup(force=True)

        was_halted = bool(self.risk.halted_until())
        halted = self.risk.update_circuit_breaker(prices, now)
        if halted and not was_halted:
            await self.notifier.breaker_tripped(str(self.risk.halted_until()), self.pf.equity(prices))
        for sym, (_, df) in frames.items():
            for r in self.shadows.resolve(sym, df, now):
                log.info("shadow %s %s -> %s", r["symbol"], r["setup_type"], r["outcome"])

        setups = 0
        if halted:
            log.warning("circuit breaker active until %s, no new entries", self.risk.halted_until())
        else:
            for sym, (ac, df) in frames.items():
                if ac == "stock" and not stocks_open:
                    continue
                setups += await self._consider(sym, ac, df, prices, now)
            if self.scanned and not setups and self.debates < self.s.max_debates_per_cycle:
                setups += await self._desk_pick(frames, stocks_open, prices, now)

        self.write_watchlist(frames, stocks_open)
        report = self.write_report(prices, now)
        self.activity.cycle(now, list(frames), failed, stocks_open, setups, bool(halted),
                            report["equity_eur"], self.scanned)
        self.pf.set_state("live_status", "idle, waiting for the next cycle")
        await self.backup.maybe_backup()
        log.info("cycle ok: %d symbols loaded%s, %d new candles scanned, %d setups flagged, equity %.2f EUR%s",
                 len(frames), f", FAILED {failed}" if failed else "", self.scanned, setups, report["equity_eur"],
                 "" if stocks_open else " (US stocks closed)")

    async def _consider(self, sym, ac, df, prices, now) -> int:
        """Returns 1 if the Scout flagged a setup on a new candle, else 0."""
        closed = df.iloc[:-1]            # the last row is the candle still forming
        if closed.empty:
            return 0
        key, bar_time = f"last_bar:{sym}", str(closed.index[-1])
        if self.pf.get_state(key) == bar_time:
            return 0                      # already looked at this candle
        self.scanned += 1

        setup = self.scout.scan(sym, ac, closed)
        if setup is None or self.pf.position_for(sym):
            self.pf.set_state(key, bar_time)
            return 0
        if not self.router.can_afford(DEBATE_CALLS):
            # Not enough free calls for a full debate. Skip it, and don't mark the candle as
            # seen, so a half-run debate can't be graded as a "pass".
            log.warning("%s %s flagged but under %d free LLM calls left today, skipping",
                        sym, setup.setup_type, DEBATE_CALLS)
            self.pf.set_state("live_status", "out of free LLM calls for today, setups skipped")
            return 0
        if self.debates >= self.s.max_debates_per_cycle:
            return 0                      # candle not marked as seen, so the next cycle picks it up
        self.pf.set_state(key, bar_time)
        await self._debate(setup, ac, prices, now)
        return 1

    async def _desk_pick(self, frames: dict, stocks_open: bool, prices: dict, now: datetime) -> int:
        """No Scout setup this hour: every few hours, debate the best-looking coin anyway.

        Gives the agents something to judge in quiet markets. It's tracked as its own setup
        type, so the scoreboard shows whether these picks work or not.
        """
        every = self.s.scout.desk_pick_hours
        if every <= 0:
            return 0
        last = self.pf.get_state("last_desk_pick")
        if last and now - datetime.fromisoformat(last) < timedelta(hours=every) - timedelta(minutes=10):
            return 0
        if not self.router.can_afford(DEBATE_CALLS + self.s.scout.desk_pick_reserve_calls):
            return 0                       # keep calls back for real Scout setups
        held = {p.symbol for p in self.pf.open_positions()}
        candidates = {sym: (ac, df.iloc[:-1]) for sym, (ac, df) in frames.items()
                      if sym not in held and (ac == "crypto" or stocks_open)}
        setup = self.scout.best_candidate(candidates)
        if setup is None:
            return 0
        self.pf.set_state("last_desk_pick", now.isoformat())
        await self._debate(setup, setup.asset_class, prices, now)
        return 1

    async def _debate(self, setup, ac, prices, now) -> None:
        sym = setup.symbol
        self.debates += 1
        self.beat()
        self.pf.set_state("live_status", f"desk debating {sym} ({setup.setup_type})")
        try:
            verdict = await asyncio.wait_for(self.ultron.deliberate(setup), self.s.debate_timeout_s)
        except asyncio.TimeoutError:
            log.warning("%s %s: debate took over %ds (LLMs too busy), dropped as HOLD",
                        sym, setup.setup_type, self.s.debate_timeout_s)
            return
        self.beat()
        votes = [asdict(v) for v in verdict.votes]
        signal, practice = verdict.signal, False
        if signal.action != "BUY":
            pick = self.practice_pick(verdict.votes)
            if pick:
                signal = Signal(sym, "BUY", pick.confidence, f"[practice] {pick.thesis}")
                practice = True
        decision = self.risk.evaluate(signal, prices[sym], prices, now, practice=practice)
        executed = False
        if decision.approved and decision.action == "open":
            self.risk.execute(decision, signal, ac, prices[sym], now, extra_meta={
                "setup_type": setup.setup_type, "regime": setup.regime, "practice": practice,
                "indicators": setup.indicators, "votes": votes,
                "council_score": verdict.council_score, "lesson_ids": verdict.lesson_ids})
            executed = True
            sl, tp = self.s.risk.stop_loss_pct, self.s.risk.take_profit_pct
            await self.notifier.trade_opened(sym, decision.notional_eur, prices[sym],
                                             prices[sym] * (1 - sl / 100), prices[sym] * (1 + tp / 100),
                                             signal.thesis, votes)
        self.shadows.record(setup, verdict.signal.action, executed, votes,
                            self.s.risk.stop_loss_pct, self.s.risk.take_profit_pct,
                            self.s.shadow_horizon_hours)
        self.activity.debate(now, setup, votes, verdict.council_score, verdict.reason,
                             decision.reason, verdict.signal.action, executed)
        log.info("%s %s [%s] -> %s | risk: %s%s", sym, setup.setup_type, setup.regime,
                 verdict.reason, decision.reason,
                 f" | OPENED{' PRACTICE' if practice else ''} {decision.notional_eur:.2f} EUR" if executed else "")
        await self.backup.maybe_backup(force=True)        # a finished debate is worth keeping

    def beat(self) -> None:
        """Tells /health the loop is alive even while a long cycle is still running."""
        self.pf.set_state("heartbeat", utcnow().isoformat())

    def practice_pick(self, votes):
        """The most confident lead that still said BUY after a HOLD verdict, if it clears the bar."""
        if not self.s.practice.enabled:
            return None
        buys = [v for v in votes if v.agent in COUNCIL and v.action == "BUY"
                and v.confidence >= self.s.practice.min_lead_confidence]
        return max(buys, key=lambda v: v.confidence) if buys else None

    def write_watchlist(self, frames: dict, stocks_open: bool) -> None:
        """What the Scout sees right now, per coin, for the dashboard's live board."""
        rules = self.s.scout
        board = {}
        for sym, (ac, df) in frames.items():
            cur = df.iloc[-1]
            if pd.isna(cur["rsi14"]) or pd.isna(cur["ema200"]):
                continue
            close = float(cur["close"])
            board[sym] = {
                "ac": ac, "price": close, "rsi": round(float(cur["rsi14"]), 1),
                "dist_ema200": round((close / float(cur["ema200"]) - 1) * 100, 2),
                "atr_pct": round(float(cur["atr_pct"]), 2),
                "vol_ratio": None if pd.isna(cur["volume_ratio"]) else round(float(cur["volume_ratio"]), 2),
                "held": bool(self.pf.position_for(sym)),
                "spark": [round(float(x), 6) for x in df["close"].iloc[-48:]],
            }
        self.pf.set_state("watchlist", json.dumps({
            "as_of": utcnow().isoformat(), "rsi_oversold": rules.rsi_oversold,
            "max_atr_pct": rules.max_atr_pct, "stocks_open": stocks_open, "coins": board}))

    async def refresh_live_prices(self) -> None:
        """Crypto prices every few seconds for the dashboard. Never trades, never raises."""
        try:
            prices = await self.data.live_prices(self.s.crypto_symbols)
            self.pf.set_state("live_prices", json.dumps({"as_of": utcnow().isoformat(), "prices": prices}))
        except Exception as e:
            log.debug("live prices failed: %s", e)

    def write_report(self, prices: dict, now: datetime) -> dict:
        positions = [{"symbol": p.symbol, "entry": p.entry_price, "stop": p.stop_price,
                      "take_profit": p.take_profit_price, "cost_eur": p.cost_eur,
                      "unrealized_eur": round(p.unrealized_pnl(prices.get(p.symbol, p.entry_price)), 4)}
                     for p in self.pf.open_positions()]
        report = {
            "as_of": now.isoformat(),
            "equity_eur": round(self.pf.equity(prices), 4),
            "halted_until": str(self.risk.halted_until() or ""),
            "performance": self.pf.metrics(),
            "open_positions": positions,
            "learning": {"lessons_stored": self.lessons.count(),
                         "agent_scoreboard": self.scores.all(),
                         "shadow_results": self.shadows.summary()},
            "llm_budget_today": self.router.usage(),
        }
        with open(self.s.report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return report


async def amain() -> None:
    import httpx

    from dashboard import serve
    from data_feed import MarketData
    from llm import build_providers

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(SETTINGS.log_path)])
    async with httpx.AsyncClient() as http:
        providers, limits = build_providers(SETTINGS.llm, http)
        if not providers:
            log.warning("No LLM keys found. Scout and shadow tracking still run, "
                        "but every decision will be HOLD.")
        log.info("Ultron online. Agents on: %s", ", ".join(providers) or "none")
        backup = Backup(http, SETTINGS.db_path)
        await backup.restore_if_missing()
        log.info("ledger backup: %s", f"on, to {backup.repo}@{backup.branch}" if backup.enabled else "off")
        notifier = Notifier(http)
        log.info("email alerts: %s", f"on, to {notifier.to}" if notifier.enabled else "off")
        bot = Bot(SETTINGS, MarketData(SETTINGS), providers, limits, notifier, backup)
        serve(SETTINGS.db_path, limits)

        async def price_loop():
            while True:
                await bot.refresh_live_prices()
                await asyncio.sleep(SETTINGS.live_price_seconds)
        price_task = asyncio.create_task(price_loop())      # noqa: F841 (kept alive by the loop)
        while True:
            try:
                await bot.cycle()
            except Exception:
                log.exception("cycle failed, carrying on")
            await asyncio.sleep(SETTINGS.cycle_seconds)


if __name__ == "__main__":
    asyncio.run(amain())
