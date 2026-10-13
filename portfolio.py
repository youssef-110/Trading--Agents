"""Virtual paper-trading ledger backed by SQLite.

All money and prices in here are EUR. Callers convert quotes (USD stocks, etc.)
before passing them in. Long-only for v1: a SELL closes an open position, it
never opens a short.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, pstdev

from config import ExecutionCosts

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL,
    starting_balance REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    cost_eur REAL NOT NULL,
    entry_fee REAL NOT NULL,
    stop_price REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    opened_at TEXT NOT NULL,
    thesis TEXT,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    cost_eur REAL NOT NULL,
    fees_eur REAL NOT NULL,
    pnl_eur REAL NOT NULL,
    return_pct REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    thesis TEXT,
    meta TEXT
);
CREATE TABLE IF NOT EXISTS equity_snapshots (
    ts TEXT NOT NULL,
    equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class InsufficientFunds(Exception):
    pass


@dataclass
class Position:
    id: int
    symbol: str
    asset_class: str
    qty: float
    entry_price: float
    cost_eur: float
    entry_fee: float
    stop_price: float
    take_profit_price: float
    opened_at: str
    thesis: str = ""
    meta: dict = field(default_factory=dict)

    def unrealized_pnl(self, price: float) -> float:
        return self.qty * price - self.cost_eur - self.entry_fee


@dataclass
class ClosedTrade:
    symbol: str
    asset_class: str
    qty: float
    entry_price: float
    exit_price: float
    pnl_eur: float
    return_pct: float
    exit_reason: str
    opened_at: str
    closed_at: str
    thesis: str = ""
    meta: dict = field(default_factory=dict)


class Portfolio:
    def __init__(self, db_path: str, starting_balance: float, costs: ExecutionCosts):
        self.costs = costs
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if self.db.execute("SELECT 1 FROM account").fetchone() is None:
            with self.db:
                self.db.execute(
                    "INSERT INTO account (id, cash, starting_balance) VALUES (1, ?, ?)",
                    (starting_balance, starting_balance),
                )
        elif (self.starting_balance != starting_balance and self.cash == self.starting_balance
              and not self.db.execute("SELECT 1 FROM trades").fetchone()
              and not self.db.execute("SELECT 1 FROM positions").fetchone()):
            # Never traded yet, so a new starting balance in config can still take effect.
            with self.db:
                self.db.execute("UPDATE account SET cash = ?, starting_balance = ?",
                                (starting_balance, starting_balance))
                self.db.execute("DELETE FROM equity_snapshots")   # old curve was at the old balance

    # ---------- basic state ----------

    @property
    def cash(self) -> float:
        return self.db.execute("SELECT cash FROM account").fetchone()[0]

    @property
    def starting_balance(self) -> float:
        return self.db.execute("SELECT starting_balance FROM account").fetchone()[0]

    def get_state(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str | None) -> None:
        with self.db:
            if value is None:
                self.db.execute("DELETE FROM state WHERE key = ?", (key,))
            else:
                self.db.execute(
                    "INSERT INTO state (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )

    # ---------- positions ----------

    @staticmethod
    def _row_to_position(row: sqlite3.Row) -> Position:
        d = dict(row)
        d["meta"] = json.loads(d["meta"]) if d["meta"] else {}
        d["thesis"] = d["thesis"] or ""
        return Position(**d)

    def open_positions(self) -> list[Position]:
        rows = self.db.execute("SELECT * FROM positions ORDER BY id").fetchall()
        return [self._row_to_position(r) for r in rows]

    def position_by_id(self, position_id: int) -> Position | None:
        row = self.db.execute("SELECT * FROM positions WHERE id = ?", (position_id,)).fetchone()
        return self._row_to_position(row) if row else None

    def position_for(self, symbol: str) -> Position | None:
        row = self.db.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()
        return self._row_to_position(row) if row else None

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus open positions marked at `prices`.

        A position with no fresh quote is marked at its entry price, which is
        optimistic, so the main loop should always pass current prices.
        """
        total = self.cash
        for p in self.open_positions():
            total += p.qty * prices.get(p.symbol, p.entry_price)
        return total

    def open_position(
        self,
        symbol: str,
        asset_class: str,
        price: float,
        notional_eur: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        thesis: str = "",
        meta: dict | None = None,
        now: datetime | None = None,
    ) -> Position:
        if price <= 0 or notional_eur <= 0:
            raise ValueError("price and notional must be positive")
        fill = price * (1 + self.costs.slippage_rate)
        fee = notional_eur * self.costs.fee_rate
        cash = self.cash
        if notional_eur + fee > cash + 1e-9:
            raise InsufficientFunds(f"need {notional_eur + fee:.4f} EUR, have {cash:.4f}")
        qty = notional_eur / fill
        stop = fill * (1 - stop_loss_pct / 100)
        tp = fill * (1 + take_profit_pct / 100)
        with self.db:
            self.db.execute("UPDATE account SET cash = ?", (cash - notional_eur - fee,))
            cur = self.db.execute(
                "INSERT INTO positions (symbol, asset_class, qty, entry_price, cost_eur, entry_fee,"
                " stop_price, take_profit_price, opened_at, thesis, meta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol, asset_class, qty, fill, notional_eur, fee, stop, tp,
                 _iso(now or utcnow()), thesis, json.dumps(meta or {})),
            )
        return self.position_by_id(cur.lastrowid)

    def close_position(
        self, position_id: int, price: float, reason: str, now: datetime | None = None
    ) -> ClosedTrade:
        p = self.position_by_id(position_id)
        if p is None:
            raise KeyError(f"no open position {position_id}")
        fill = price * (1 - self.costs.slippage_rate)
        proceeds = p.qty * fill
        exit_fee = proceeds * self.costs.fee_rate
        pnl = proceeds - exit_fee - p.cost_eur - p.entry_fee
        ret = pnl / (p.cost_eur + p.entry_fee) * 100
        closed_at = _iso(now or utcnow())
        with self.db:
            self.db.execute("UPDATE account SET cash = cash + ?", (proceeds - exit_fee,))
            self.db.execute("DELETE FROM positions WHERE id = ?", (p.id,))
            self.db.execute(
                "INSERT INTO trades (symbol, asset_class, qty, entry_price, exit_price, cost_eur,"
                " fees_eur, pnl_eur, return_pct, exit_reason, opened_at, closed_at, thesis, meta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (p.symbol, p.asset_class, p.qty, p.entry_price, fill, p.cost_eur,
                 p.entry_fee + exit_fee, pnl, ret, reason, p.opened_at, closed_at,
                 p.thesis, json.dumps(p.meta)),
            )
        return ClosedTrade(p.symbol, p.asset_class, p.qty, p.entry_price, fill, pnl, ret,
                           reason, p.opened_at, closed_at, p.thesis, p.meta)

    # ---------- history & metrics ----------

    def record_equity(self, equity: float, now: datetime | None = None) -> None:
        with self.db:
            self.db.execute("INSERT INTO equity_snapshots (ts, equity) VALUES (?, ?)",
                            (_iso(now or utcnow()), equity))

    def equity_history(self, since: datetime | None = None) -> list[tuple[datetime, float]]:
        if since is None:
            rows = self.db.execute("SELECT ts, equity FROM equity_snapshots ORDER BY ts").fetchall()
        else:
            rows = self.db.execute(
                "SELECT ts, equity FROM equity_snapshots WHERE ts >= ? ORDER BY ts", (_iso(since),)
            ).fetchall()
        return [(_parse(r[0]), r[1]) for r in rows]

    def last_stop_out(self, symbol: str) -> datetime | None:
        row = self.db.execute(
            "SELECT closed_at FROM trades WHERE symbol = ? AND exit_reason = 'stop_loss' "
            "ORDER BY id DESC LIMIT 1", (symbol,)).fetchone()
        return _parse(row[0]) if row else None

    def trades(self) -> list[dict]:
        rows = self.db.execute("SELECT * FROM trades ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def metrics(self) -> dict:
        trades = self.trades()
        n = len(trades)
        returns = [t["return_pct"] for t in trades]
        wins = sum(1 for t in trades if t["pnl_eur"] > 0)

        # Per-trade Sharpe-style ratio. Not annualised, and meaningless with
        # only a handful of trades, so treat it as a rough signal at best.
        per_trade_sharpe = None
        if n >= 2 and pstdev(returns) > 0:
            per_trade_sharpe = mean(returns) / pstdev(returns)

        peak, max_dd = -math.inf, 0.0
        for _, eq in self.equity_history():
            peak = max(peak, eq)
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak)

        return {
            "cash_eur": round(self.cash, 4),
            "trades": n,
            "win_rate": wins / n if n else None,
            "total_pnl_eur": round(sum(t["pnl_eur"] for t in trades), 4),
            "total_fees_eur": round(sum(t["fees_eur"] for t in trades), 4),
            "avg_return_pct": mean(returns) if returns else None,
            "per_trade_sharpe": per_trade_sharpe,
            "max_drawdown_pct": max_dd * 100,
        }
