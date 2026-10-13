"""How the desk learns.

LLMs don't change their weights from experience, so "learning" here means
three concrete things, all stored in SQLite:

1. Lessons. The Historian writes a post-mortem for every closed trade. Before
   each new decision, the most relevant past lessons (same setup, same market
   regime, same symbol) go into the agents' prompts.

2. Shadow tracking. EVERY setup the Scout flags is followed to see whether it
   would have hit +4% or -2% first, including the ones Ultron passed on. That
   means the desk learns from ~10x more outcomes than it actually trades, and
   it learns what it missed, not just what it lost.

3. Agent scoreboard. Each shadow outcome grades every agent's vote. An agent
   that keeps calling BUY on losers gets less weight in the council, and Ultron
   sees everyone's record when it decides.

Shadow stats also answer the most important question: does the AI council
beat simply taking every Scout setup? If not, the LLMs aren't adding value.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT, asset_class TEXT, setup_type TEXT, regime TEXT,
    outcome TEXT, return_pct REAL,
    lesson TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shadow_setups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL, asset_class TEXT NOT NULL,
    setup_type TEXT NOT NULL, regime TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    entry_price REAL NOT NULL, stop_price REAL NOT NULL, tp_price REAL NOT NULL,
    expires_at TEXT NOT NULL,
    final_action TEXT NOT NULL, executed INTEGER NOT NULL,
    votes TEXT NOT NULL,
    outcome TEXT, resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS debates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL, setup_type TEXT NOT NULL, regime TEXT NOT NULL,
    price_eur REAL NOT NULL,
    votes TEXT NOT NULL,
    council_score REAL, reason TEXT, risk_reason TEXT,
    final_action TEXT, executed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    loaded TEXT, failed TEXT, stocks_open INTEGER,
    setups INTEGER NOT NULL DEFAULT 0, halted INTEGER NOT NULL DEFAULT 0,
    equity REAL
);
CREATE TABLE IF NOT EXISTS emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, subject TEXT NOT NULL, ok INTEGER NOT NULL, detail TEXT
);
CREATE TABLE IF NOT EXISTS agent_scores (
    agent TEXT PRIMARY KEY,
    calls INTEGER NOT NULL DEFAULT 0,
    correct INTEGER NOT NULL DEFAULT 0
);
"""


def init_memory(db: sqlite3.Connection) -> None:
    db.executescript(SCHEMA)
    try:      # older databases don't have this column yet
        db.execute("ALTER TABLE cycles ADD COLUMN checked INTEGER NOT NULL DEFAULT 0")
        db.commit()
    except sqlite3.OperationalError:
        pass


def regime(ind: dict) -> str:
    """Coarse market-state label used to match lessons to situations."""
    close, ema200, atr_pct = ind.get("close"), ind.get("ema200"), ind.get("atr_pct")
    trend = "uptrend" if close and ema200 and close > ema200 else "downtrend"
    vol = "volatile" if (atr_pct or 0) >= 0.8 else "calm"
    return f"{trend}/{vol}"


class LessonStore:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        init_memory(db)

    def add(self, symbol, asset_class, setup_type, regime_, outcome, return_pct, lesson,
            now: datetime) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO lessons (created_at, symbol, asset_class, setup_type, regime, outcome,"
                " return_pct, lesson) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (now.isoformat(), symbol, asset_class, setup_type, regime_, outcome, return_pct, lesson))
        return cur.lastrowid

    def relevant(self, symbol: str, asset_class: str, setup_type: str, regime_: str,
                 k: int = 3) -> list[dict]:
        rows = [dict(r) for r in self.db.execute(
            "SELECT * FROM lessons ORDER BY id DESC LIMIT 300").fetchall()]
        scored = []
        for r in rows:
            s = (3 * (r["setup_type"] == setup_type) + 2 * (r["regime"] == regime_)
                 + 2 * (r["symbol"] == symbol) + 1 * (r["asset_class"] == asset_class)
                 + 1 * (r["outcome"] == "loss"))      # mistakes are worth more
            if s >= 3:
                scored.append((s, r["id"], r))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [r for _, _, r in scored[:k]]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]


class Scoreboard:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        init_memory(db)

    def record(self, votes: list[dict], outcome: str) -> None:
        if outcome not in ("win", "loss"):
            return
        with self.db:
            for v in votes:
                correct = int((v["action"] == "BUY") == (outcome == "win"))
                self.db.execute(
                    "INSERT INTO agent_scores (agent, calls, correct) VALUES (?, 1, ?) "
                    "ON CONFLICT(agent) DO UPDATE SET calls = calls + 1, correct = correct + ?",
                    (v["agent"], correct, correct))

    def reliability(self, agent: str) -> float:
        """Laplace-smoothed hit rate: starts at 0.5 and moves with evidence."""
        row = self.db.execute("SELECT calls, correct FROM agent_scores WHERE agent = ?",
                              (agent,)).fetchone()
        calls, correct = (row[0], row[1]) if row else (0, 0)
        return (correct + 1) / (calls + 2)

    def all(self) -> dict:
        out = {}
        for agent, calls, correct in self.db.execute("SELECT agent, calls, correct FROM agent_scores"):
            out[agent] = {"graded_calls": calls, "correct": correct,
                          "reliability": round((correct + 1) / (calls + 2), 3)}
        return out


class ShadowTracker:
    def __init__(self, db: sqlite3.Connection, scoreboard: Scoreboard):
        self.db = db
        self.scoreboard = scoreboard
        init_memory(db)

    def record(self, setup, final_action: str, executed: bool, votes: list[dict],
               stop_loss_pct: float, take_profit_pct: float, horizon_hours: int) -> int:
        price = setup.price_eur
        expires = pd.Timestamp(setup.bar_time) + timedelta(hours=horizon_hours)
        with self.db:
            cur = self.db.execute(
                "INSERT INTO shadow_setups (symbol, asset_class, setup_type, regime, bar_time,"
                " entry_price, stop_price, tp_price, expires_at, final_action, executed, votes)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (setup.symbol, setup.asset_class, setup.setup_type, setup.regime, setup.bar_time,
                 price, price * (1 - stop_loss_pct / 100), price * (1 + take_profit_pct / 100),
                 expires.isoformat(), final_action, int(executed), json.dumps(votes)))
        return cur.lastrowid

    def resolve(self, symbol: str, df: pd.DataFrame, now: datetime) -> list[dict]:
        """Walk bars after each pending setup and see which level got hit first."""
        pending = self.db.execute(
            "SELECT * FROM shadow_setups WHERE symbol = ? AND outcome IS NULL", (symbol,)).fetchall()
        resolved = []
        for r in pending:
            after = df[df.index > pd.Timestamp(r["bar_time"])]
            outcome = None
            for _, bar in after.iterrows():
                if bar["low"] <= r["stop_price"]:      # same-bar tie counts as a loss
                    outcome = "loss"
                    break
                if bar["high"] >= r["tp_price"]:
                    outcome = "win"
                    break
            if outcome is None and pd.Timestamp(now) >= pd.Timestamp(r["expires_at"]):
                outcome = "expired"
            if outcome:
                with self.db:
                    self.db.execute("UPDATE shadow_setups SET outcome = ?, resolved_at = ? WHERE id = ?",
                                    (outcome, now.isoformat(), r["id"]))
                self.scoreboard.record(json.loads(r["votes"]), outcome)
                resolved.append({**dict(r), "outcome": outcome})
        return resolved

    def _stats(self, where: str = "", params: tuple = ()) -> dict:
        rows = self.db.execute(
            f"SELECT outcome, COUNT(*) FROM shadow_setups WHERE outcome IS NOT NULL {where} "
            "GROUP BY outcome", params).fetchall()
        c = {o: n for o, n in rows}
        wins, losses = c.get("win", 0), c.get("loss", 0)
        decided = wins + losses
        return {"wins": wins, "losses": losses, "expired": c.get("expired", 0),
                "win_rate": round(wins / decided, 3) if decided else None}

    def setup_stats(self, setup_type: str, regime_: str) -> dict:
        return {"this_setup_any_regime": self._stats("AND setup_type = ?", (setup_type,)),
                "this_setup_this_regime": self._stats("AND setup_type = ? AND regime = ?",
                                                      (setup_type, regime_))}

    def summary(self) -> dict:
        pending = self.db.execute(
            "SELECT COUNT(*) FROM shadow_setups WHERE outcome IS NULL").fetchone()[0]
        return {
            "pending": pending,
            "every_scout_setup (baseline)": self._stats(),
            "ultron_said_buy": self._stats("AND final_action = 'BUY'"),
            "ultron_passed": self._stats("AND final_action != 'BUY'"),
            "note": "Break-even win rate with 2%/4% exits and fees is about 38%.",
        }


class ActivityLog:
    """What the desk did, kept for the dashboard. Nothing here feeds back into decisions."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        init_memory(db)

    def debate(self, now: datetime, setup, verdict_votes: list[dict], council_score: float,
               reason: str, risk_reason: str, final_action: str, executed: bool) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO debates (ts, symbol, setup_type, regime, price_eur, votes, council_score,"
                " reason, risk_reason, final_action, executed) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (now.isoformat(), setup.symbol, setup.setup_type, setup.regime, setup.price_eur,
                 json.dumps(verdict_votes), council_score, reason, risk_reason, final_action,
                 int(executed)))

    def email(self, now: datetime, subject: str, ok: bool, detail: str = "") -> None:
        with self.db:
            self.db.execute("INSERT INTO emails (ts, subject, ok, detail) VALUES (?,?,?,?)",
                            (now.isoformat(), subject[:200], int(ok), detail[:300]))

    def cycle(self, now: datetime, loaded: list, failed: list, stocks_open: bool, setups: int,
              halted: bool, equity: float, checked: int = 0) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO cycles (ts, loaded, failed, stocks_open, setups, halted, equity, checked)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (now.isoformat(), ",".join(loaded), ",".join(failed), int(stocks_open), setups,
                 int(halted), equity, checked))
            self.db.execute("DELETE FROM cycles WHERE id <= (SELECT MAX(id) FROM cycles) - 2000")
