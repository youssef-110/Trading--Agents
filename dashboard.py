"""Read-only web dashboard for the desk.

Runs inside the bot process on a background thread (standard library only), so one
`python main.py` gives you the bot and the page. It only reads the SQLite file and
never touches the trading loop.

  http://localhost:8000          default
  PORT=10000                     what Render sets; binds 0.0.0.0 when PORT is set
  DASHBOARD_TOKEN=somesecret     if set, the page needs ?key=somesecret

Exposed to the network (PORT set) without a token, the dashboard refuses to start.
Only the bot's loopback address is allowed to run open.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import backup
from agents import SUBS
from config import SETTINGS
from llm import quota_day
from notify import mask, send_test_email

log = logging.getLogger("dashboard")
START = time.time()
STALE_AFTER_S = 20 * 60
PAGE = Path(__file__).with_name("dashboard.html")
ULTRON_PAGE = Path(__file__).with_name("ultron.html")


def _rows(db: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    try:
        return [dict(r) for r in db.execute(sql, params).fetchall()]
    except sqlite3.OperationalError:       # table doesn't exist yet on a brand-new database
        return []


def build_state(db_path: str, limits: dict | None = None) -> dict:
    db = sqlite3.connect(db_path, timeout=5)
    db.row_factory = sqlite3.Row
    try:
        state = {r["key"]: r["value"] for r in _rows(db, "SELECT key, value FROM state")}
        acct = (_rows(db, "SELECT cash, starting_balance FROM account") or [{}])[0]
        cycles = _rows(db, "SELECT * FROM cycles ORDER BY id DESC LIMIT 20")
        trades = _rows(db, "SELECT id, symbol, entry_price, exit_price, pnl_eur, return_pct,"
                           " exit_reason, opened_at, closed_at, thesis, meta FROM trades ORDER BY id DESC LIMIT 30")
        wins = sum(1 for t in _rows(db, "SELECT pnl_eur FROM trades") if t["pnl_eur"] > 0)
        n_trades = (_rows(db, "SELECT COUNT(*) AS n FROM trades") or [{"n": 0}])[0]["n"]
        pnl = (_rows(db, "SELECT COALESCE(SUM(pnl_eur),0) AS p FROM trades") or [{"p": 0}])[0]["p"]
        positions = _rows(db, "SELECT symbol, qty, entry_price, cost_eur, stop_price,"
                              " take_profit_price, opened_at, thesis, meta FROM positions")
        for row in positions + trades:                 # keep only the practice flag, not the whole vote dump
            try:
                row["practice"] = bool(json.loads(row.pop("meta") or "{}").get("practice"))
            except (TypeError, ValueError):
                row["practice"] = False

        debates = _rows(db, "SELECT * FROM debates ORDER BY id DESC LIMIT 25")
        for d in debates:
            d["votes"] = json.loads(d["votes"])

        scores = {r["agent"]: r for r in _rows(db, "SELECT agent, calls, correct FROM agent_scores")}
        won = {}                                   # how often each specialist won its lead's pick
        for d in debates:
            for v in d["votes"]:
                if v.get("winner"):
                    won[v["winner"]] = won.get(v["winner"], 0) + 1

        def agent(name: str, lens: str = "") -> dict:
            s = scores.get(name, {"calls": 0, "correct": 0})
            return {"name": name, "lens": lens, "graded": s["calls"], "correct": s["correct"],
                    "reliability": round((s["correct"] + 1) / (s["calls"] + 2), 3),
                    "times_picked": won.get(name, 0)}

        tree = {"manager": agent("ultron"), "leads": [
            {**agent(lead), "subs": [agent(n, lens) for n, lens in subs]}
            for lead, subs in SUBS.items()]}

        shadow = {}
        for label, where in (("every_setup", ""), ("ultron_buy", "AND final_action='BUY'"),
                             ("ultron_passed", "AND final_action!='BUY'")):
            c = {r["outcome"]: r["n"] for r in _rows(
                db, f"SELECT outcome, COUNT(*) AS n FROM shadow_setups WHERE outcome IS NOT NULL {where}"
                    " GROUP BY outcome")}
            w, l = c.get("win", 0), c.get("loss", 0)
            shadow[label] = {"wins": w, "losses": l, "expired": c.get("expired", 0),
                             "win_rate": round(w / (w + l), 3) if w + l else None}
        shadow["pending"] = (_rows(db, "SELECT COUNT(*) AS n FROM shadow_setups WHERE outcome IS NULL")
                             or [{"n": 0}])[0]["n"]

        used = {r["provider"]: r["calls"] for r in _rows(
            db, "SELECT provider, calls FROM llm_calls WHERE day = ?", (quota_day(),))}
        llm = {p: {"used": used.get(p, 0), "limit": lim[0]} for p, lim in (limits or {}).items()}
        for p, n in used.items():
            llm.setdefault(p, {"used": n, "limit": None})

        day_ago = datetime.now(timezone.utc).timestamp() - 86400
        recent = [c for c in _rows(db, "SELECT ts, setups, checked FROM cycles ORDER BY id DESC LIMIT 400")
                  if datetime.fromisoformat(c["ts"]).timestamp() > day_ago]
        scout = {"candles_checked_24h": sum(c["checked"] or 0 for c in recent),
                 "setups_flagged_24h": sum(c["setups"] or 0 for c in recent),
                 "cycles_24h": len(recent)}
        email = {"configured": bool(os.getenv("RESEND_API_KEY") and os.getenv("EMAIL_TO")),
                 "to": mask(os.getenv("EMAIL_TO", "")),
                 "recent": _rows(db, "SELECT ts, subject, ok, detail FROM emails ORDER BY id DESC LIMIT 8")}

        equity = cycles[0]["equity"] if cycles and cycles[0]["equity"] is not None else acct.get("cash")

        def blob(key: str) -> dict:
            try:
                return json.loads(state.get(key) or "{}")
            except ValueError:
                return {}
        live = blob("live_prices")
        for pos in positions:                          # mark open positions at the freshest price we have
            px = (live.get("prices") or {}).get(pos["symbol"])
            pos["last"] = px
            pos["unrealized_eur"] = None if px is None else round(pos["qty"] * px - pos["cost_eur"], 4)
        if acct.get("cash") is not None and all(p["last"] is not None for p in positions):
            equity = round(acct["cash"] + sum(p["qty"] * p["last"] for p in positions), 4)

        by_setup = {}
        for r in _rows(db, "SELECT setup_type, outcome, COUNT(*) AS n FROM shadow_setups"
                           " WHERE outcome IS NOT NULL GROUP BY setup_type, outcome"):
            d = by_setup.setdefault(r["setup_type"], {"win": 0, "loss": 0, "expired": 0})
            d[r["outcome"]] = r["n"]
        for d in by_setup.values():
            d["win_rate"] = round(d["win"] / (d["win"] + d["loss"]), 3) if d["win"] + d["loss"] else None
        day_iso = datetime.fromtimestamp(day_ago, timezone.utc).isoformat()
        funnel = {"checked": scout["candles_checked_24h"], "flagged": scout["setups_flagged_24h"],
                  "debated": (_rows(db, "SELECT COUNT(*) AS n FROM debates WHERE ts > ?", (day_iso,)) or [{"n": 0}])[0]["n"],
                  "opened": (_rows(db, "SELECT COUNT(*) AS n FROM debates WHERE ts > ? AND executed = 1",
                                   (day_iso,)) or [{"n": 0}])[0]["n"]}
        return {
            "now": datetime.now(timezone.utc).isoformat(),
            "status": state.get("live_status", "starting up"),
            "halted_until": state.get("halted_until") or "",
            "account": {"starting": acct.get("starting_balance"), "cash": acct.get("cash"),
                        "equity": equity, "trades": n_trades, "wins": wins,
                        "realised_pnl": round(pnl, 4)},
            "positions": positions, "trades": trades, "tree": tree, "debates": debates,
            "cycles": cycles, "shadow": shadow, "llm": llm, "scout": scout, "email": email,
            "watchlist": blob("watchlist"), "live_prices": live, "by_setup": by_setup, "funnel": funnel,
            "cycle_seconds": SETTINGS.cycle_seconds,
            "practice": {"enabled": SETTINGS.practice.enabled, "size_pct": SETTINGS.practice.position_pct,
                         "max_open": SETTINGS.practice.max_open},
            "risk": {"stop_pct": SETTINGS.risk.stop_loss_pct, "target_pct": SETTINGS.risk.take_profit_pct,
                     "max_positions": SETTINGS.risk.max_open_positions},
            "equity_curve": [[r["ts"], r["equity"]] for r in reversed(_rows(
                db, "SELECT ts, equity FROM equity_snapshots ORDER BY rowid DESC LIMIT 600"))],
            "lessons": _rows(db, "SELECT created_at, symbol, setup_type, outcome, return_pct, lesson"
                                 " FROM lessons ORDER BY id DESC LIMIT 12"),
        }
    finally:
        db.close()


LOGIN_PAGE = b"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Ultron</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; min-height:100vh; display:grid; place-items:center; background:#0b0d10; color:#dde3ea;
         font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
  form { width:min(92vw,380px); display:grid; gap:12px; }
  h1 { margin:0; font-size:26px; letter-spacing:.14em; }
  p { margin:0; color:#8e98a5; }
  input { font:inherit; padding:11px 12px; border-radius:6px; border:1px solid #2a313a; background:#12151a; color:inherit; }
  input:focus { outline:2px solid #ff4a2e; outline-offset:1px; }
  button { font:inherit; font-weight:600; padding:11px; border:0; border-radius:6px; background:#ff4a2e; color:#fff; cursor:pointer; }
  .bad { color:#ff7a66; }
</style></head><body>
<form id="f"><h1>ULTRON</h1>
<p id="msg">Enter the dashboard key (the DASHBOARD_TOKEN you set in Render).</p>
<input id="k" type="password" autocomplete="current-password" aria-label="Dashboard key" autofocus required>
<button type="submit">Open</button></form>
<script>
const had = new URLSearchParams(location.search).has("key");
if (had) { const m = document.getElementById("msg"); m.textContent = "That key didn't match. Check it in Render > Environment and try again."; m.className = "bad"; }
document.getElementById("f").addEventListener("submit", e => { e.preventDefault();
  location.href = location.pathname + "?key=" + encodeURIComponent(document.getElementById("k").value.trim()); });
</script></body></html>"""


def health(db_path: str) -> tuple[bool, str]:
    """Unhealthy when the trading loop has gone quiet, so an uptime pinger can email you."""
    note = f" | backup {backup.STATUS['text']}"
    cycle = beat = None
    try:
        db = sqlite3.connect(db_path, timeout=5)
        try:
            row = db.execute("SELECT ts FROM cycles ORDER BY id DESC LIMIT 1").fetchone()
            cycle = row[0] if row else None
            row = db.execute("SELECT value FROM state WHERE key = 'heartbeat'").fetchone()
            beat = row[0] if row else None
        finally:
            db.close()
    except sqlite3.Error:
        pass
    # The heartbeat moves during a long cycle (a debate can take minutes), so a busy loop
    # isn't mistaken for a dead one and restarted by Render.
    ages = [time.time() - datetime.fromisoformat(ts).timestamp() for ts in (cycle, beat) if ts]
    if not ages:
        return (time.time() - START < STALE_AFTER_S), "starting, no cycle yet" + note
    age = min(ages)
    what = f"last cycle {int(time.time() - datetime.fromisoformat(cycle).timestamp())}s ago" if cycle \
        else "first cycle running"
    return age < STALE_AFTER_S, f"{what}, alive {int(age)}s ago{note}"


def bind_settings(env: dict) -> tuple[str, int, str] | None:
    """(host, port, token), or None when it would be public with no token."""
    port_env = env.get("PORT")
    port = int(port_env or env.get("DASHBOARD_PORT", "8000"))
    host = "0.0.0.0" if port_env else "127.0.0.1"
    token = env.get("DASHBOARD_TOKEN", "")
    if host != "127.0.0.1" and not token:
        return None
    return host, port, token


def serve(db_path: str, limits: dict | None = None) -> ThreadingHTTPServer | None:
    settings = bind_settings(os.environ)
    if settings is None:
        log.error("dashboard NOT started: PORT is set so it would be public, but DASHBOARD_TOKEN "
                  "is empty. Set DASHBOARD_TOKEN to turn it on. The bot itself keeps running.")
        return None
    host, port, token = settings

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/health":                      # for uptime pingers; no private data
                ok, msg = health(db_path)
                return self._send(200 if ok else 503, msg.encode(), "text/plain")
            if token and not hmac.compare_digest(
                    parse_qs(url.query).get("key", [""])[0].encode(), token.encode()):
                if url.path in ("/", "/index.html", "/ultron"):
                    return self._send(401, LOGIN_PAGE, "text/html; charset=utf-8")
                return self._send(401, b"needs ?key=...", "text/plain")
            if url.path == "/api/state":
                try:
                    body = json.dumps(build_state(db_path, limits), default=str).encode()
                except Exception:
                    log.exception("dashboard state failed")
                    return self._send(500, b"{}", "application/json")
                return self._send(200, body, "application/json")
            if url.path == "/ultron":
                return self._send(200, ULTRON_PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path != "/api/test-email":
                return self._send(404, b"not found", "text/plain")
            # A header a random website can't add to a cross-site request, plus the same token as GET.
            if self.headers.get("X-Ultron") != "1":
                return self._send(403, b"missing header", "text/plain")
            if token and not hmac.compare_digest(
                    parse_qs(url.query).get("key", [""])[0].encode(), token.encode()):
                return self._send(401, b"needs ?key=...", "text/plain")
            return self._send(200, json.dumps(send_test_email(db_path)).encode(), "application/json")

        def log_message(self, *args):                       # keep trader.log readable
            pass

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        log.warning("dashboard not started (port %s): %s", port, e)
        return None
    threading.Thread(target=server.serve_forever, daemon=True, name="dashboard").start()
    log.info("dashboard on http://%s:%d%s", "localhost" if host == "127.0.0.1" else host, port,
             "  (token required)" if token else "")
    return server
