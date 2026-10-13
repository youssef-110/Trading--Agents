"""Deterministic risk layer. The models propose, this module decides.

From a model signal only the ticker, action and confidence are used. Stop-loss,
take-profit and size always come from config. The model's suggested SL/TP are
kept on the signal for logging and post-mortems, never for execution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import PracticeMode, RiskLimits
from portfolio import ClosedTrade, Portfolio, utcnow

VALID_ACTIONS = {"BUY", "SELL", "HOLD"}
HALT_KEY = "halted_until"


@dataclass
class Signal:
    ticker: str
    action: str
    confidence: float
    thesis: str = ""
    suggested_stop_loss_pct: float | None = None
    suggested_take_profit_pct: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Signal":
        """Strict parse of model JSON. Anything malformed raises ValueError."""
        action = str(d.get("action", "")).strip().upper()
        if action not in VALID_ACTIONS:
            raise ValueError(f"bad action: {d.get('action')!r}")
        conf = float(d["confidence"])
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence out of range: {conf}")
        ticker = str(d["ticker"]).strip()
        if not ticker:
            raise ValueError("empty ticker")

        def opt(key):
            v = d.get(key)
            return None if v is None else float(v)

        return cls(ticker, action, conf, str(d.get("thesis", "")),
                   opt("suggested_stop_loss_pct"), opt("suggested_take_profit_pct"))


@dataclass
class Decision:
    approved: bool
    action: str            # "open" | "close" | "none"
    reason: str
    notional_eur: float = 0.0
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    position_id: int | None = None


@dataclass
class Bar:
    """Latest price info for one symbol.

    low/high should cover only the time since the last exit check, otherwise a
    wick from before the position existed could trigger a stop.
    """
    last: float
    low: float | None = None
    high: float | None = None


class RiskManager:
    def __init__(self, portfolio: Portfolio, limits: RiskLimits, practice: PracticeMode | None = None):
        self.portfolio = portfolio
        self.limits = limits
        self.practice = practice or PracticeMode(enabled=False)

    # ---------- circuit breaker ----------

    def halted_until(self) -> datetime | None:
        v = self.portfolio.get_state(HALT_KEY)
        return datetime.fromisoformat(v) if v else None

    def is_halted(self, now: datetime | None = None) -> bool:
        until = self.halted_until()
        return until is not None and (now or utcnow()) < until

    def update_circuit_breaker(self, prices: dict[str, float], now: datetime | None = None) -> bool:
        """Snapshot equity and trip the breaker if we're down too far from the 24h high.

        The halt is stored in the DB, so restarting the bot doesn't reset it.
        Returns True while trading is halted.
        """
        now = now or utcnow()
        equity = self.portfolio.equity(prices)
        self.portfolio.record_equity(equity, now)
        if self.is_halted(now):
            return True

        window = self.portfolio.equity_history(since=now - timedelta(hours=24))
        peak = max(eq for _, eq in window)
        drop_pct = (peak - equity) / peak * 100 if peak > 0 else 0.0
        if drop_pct >= self.limits.daily_loss_limit_pct:
            until = now + timedelta(hours=self.limits.circuit_breaker_pause_hours)
            self.portfolio.set_state(HALT_KEY, until.isoformat())
            return True
        return False

    # ---------- entry / exit decisions ----------

    def evaluate(self, signal: Signal, price_eur: float, prices: dict[str, float],
                 now: datetime | None = None, practice: bool = False) -> Decision:
        """`practice=True` is a small learning trade: smaller size, lower confidence bar, own cap."""
        if practice and not self.practice.enabled:
            return Decision(False, "none", "practice mode is off")
        if signal.action == "HOLD":
            return Decision(False, "none", "model says HOLD")

        existing = self.portfolio.position_for(signal.ticker)

        if signal.action == "SELL":
            # Exits are always allowed, even while halted, because they reduce risk.
            if existing:
                return Decision(True, "close", "model exit signal", position_id=existing.id)
            return Decision(False, "none", "nothing to sell (long-only)")

        # BUY
        if self.is_halted(now):
            return Decision(False, "none", f"circuit breaker active until {self.halted_until()}")
        if existing:
            return Decision(False, "none", "already holding this symbol")
        stopped = self.portfolio.last_stop_out(signal.ticker)
        if stopped and (now or utcnow()) - stopped < timedelta(hours=self.limits.reentry_cooldown_hours):
            return Decision(False, "none", f"cooling off after stop-loss at {stopped:%Y-%m-%d %H:%M} UTC")
        if len(self.portfolio.open_positions()) >= self.limits.max_open_positions:
            return Decision(False, "none", "max open positions reached")
        min_conf = self.practice.min_lead_confidence if practice else self.limits.min_confidence
        if signal.confidence < min_conf:
            return Decision(False, "none", f"confidence {signal.confidence:.2f} below threshold")
        if practice and sum(1 for p in self.portfolio.open_positions()
                            if p.meta.get("practice")) >= self.practice.max_open:
            return Decision(False, "none", "max practice trades open")
        if price_eur <= 0:
            return Decision(False, "none", "invalid price")

        equity = self.portfolio.equity(prices)
        notional = equity * (self.practice.position_pct if practice else self.limits.max_position_pct)
        affordable = self.portfolio.cash / (1 + self.portfolio.costs.fee_rate)
        notional = math.floor(min(notional, affordable) * 100) / 100  # whole cents, rounded down
        if notional < self.limits.min_order_eur:
            return Decision(False, "none", f"order size {notional:.2f} EUR below minimum")

        return Decision(True, "open", "practice trade, passed all checks" if practice else "passed all checks",
                        notional_eur=notional,
                        stop_loss_pct=self.limits.stop_loss_pct,
                        take_profit_pct=self.limits.take_profit_pct)

    def execute(self, decision: Decision, signal: Signal, asset_class: str, price_eur: float,
                now: datetime | None = None, extra_meta: dict | None = None):
        """Apply an approved decision to the ledger. Returns the Position or ClosedTrade."""
        if not decision.approved:
            return None
        if decision.action == "open":
            meta = {"confidence": signal.confidence,
                    "model_sl": signal.suggested_stop_loss_pct,
                    "model_tp": signal.suggested_take_profit_pct,
                    **(extra_meta or {})}
            return self.portfolio.open_position(
                signal.ticker, asset_class, price_eur, decision.notional_eur,
                decision.stop_loss_pct, decision.take_profit_pct, thesis=signal.thesis,
                meta=meta, now=now)
        if decision.action == "close":
            return self.portfolio.close_position(decision.position_id, price_eur, "model_exit", now)
        return None

    def check_exits(self, bars: dict[str, Bar], now: datetime | None = None) -> list[ClosedTrade]:
        """Enforce hard stop-loss and take-profit on every open position.

        If one bar touches both levels we assume the stop hit first, since we
        can't know the order and it's the conservative guess. If price gapped
        through the stop we fill at the worse last price, not at the stop.
        """
        closed = []
        for p in self.portfolio.open_positions():
            bar = bars.get(p.symbol)
            if bar is None:
                continue
            low = bar.low if bar.low is not None else bar.last
            high = bar.high if bar.high is not None else bar.last
            if low <= p.stop_price:
                fill = p.stop_price if bar.last >= p.stop_price else bar.last
                closed.append(self.portfolio.close_position(p.id, fill, "stop_loss", now))
            elif high >= p.take_profit_price:
                closed.append(self.portfolio.close_position(
                    p.id, p.take_profit_price, "take_profit", now))
        return closed
