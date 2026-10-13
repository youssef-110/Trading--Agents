"""Email alerts through Resend.

One email when the desk opens a paper trade, one when a trade closes (profit or
loss), one when the circuit breaker trips. Sending never raises: if Resend is down
or the key is wrong, the bot logs a warning and carries on trading.

Setup, in .env:
  RESEND_API_KEY=re_...
  EMAIL_TO=you@example.com
  EMAIL_FROM=Ultron <onboarding@resend.dev>     (optional; Resend's test sender only
                                                  delivers to the address that owns the account)
"""
from __future__ import annotations

import html
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

import httpx

log = logging.getLogger("notify")
URL = "https://api.resend.com/emails"


def _e(x) -> str:
    return html.escape(str(x))


class Notifier:
    def __init__(self, http: httpx.AsyncClient | None, api_key: str | None = None,
                 to: str | None = None, sender: str | None = None):
        self.http = http
        self.api_key = api_key if api_key is not None else os.getenv("RESEND_API_KEY", "")
        self.to = to if to is not None else os.getenv("EMAIL_TO", "")
        self.sender = sender or os.getenv("EMAIL_FROM", "Ultron <onboarding@resend.dev>")
        self.sent = 0
        self.on_result = None            # optional callback(subject, ok, detail) for the dashboard log

    @property
    def enabled(self) -> bool:
        return bool(self.http and self.api_key and self.to)

    async def send(self, subject: str, body_html: str) -> bool:
        if not self.enabled:
            return False
        ok, detail = False, ""
        try:
            r = await self.http.post(
                URL, headers={"Authorization": f"Bearer {self.api_key}"},
                json={"from": self.sender, "to": [self.to], "subject": subject, "html": body_html},
                timeout=20)
            if r.status_code >= 300:
                detail = f"HTTP {r.status_code} {r.text[:200]}"
                log.warning("email not sent: %s", detail)
            else:
                ok, detail = True, "accepted by Resend"
                self.sent += 1
        except Exception as e:  # an alert must never break the trading loop
            detail = str(e)[:200]
            log.warning("email not sent: %s", detail)
        if self.on_result:
            try:
                self.on_result(subject, ok, detail)
            except Exception:
                log.exception("couldn't log the email result")
        return ok

    # ---------- the three events ----------

    async def trade_opened(self, symbol: str, cost_eur: float, entry: float, stop: float,
                           target: float, thesis: str, votes: list[dict]) -> bool:
        buys = [v["agent"] for v in votes if v.get("action") == "BUY" and "." not in v["agent"]]
        body = (f"<h2>Bought {_e(symbol)} (paper)</h2>"
                f"<p>Spent <b>€{cost_eur:.2f}</b> at {entry:.4f}. Stop {stop:.4f} (-2%), "
                f"target {target:.4f} (+4%).</p>"
                f"<p><b>Why:</b> {_e(thesis)}</p>"
                f"<p>Leads that voted BUY: {_e(', '.join(buys) or 'none')}.</p>"
                f"<p style='color:#888'>Paper money only. Nothing real was traded.</p>")
        return await self.send(f"Ultron bought {symbol} for €{cost_eur:.2f}", body)

    async def trade_closed(self, symbol: str, pnl_eur: float, return_pct: float, reason: str,
                           entry: float, exit_: float, equity: float | None = None) -> bool:
        word = "Profit" if pnl_eur > 0 else "Loss"
        why = {"stop_loss": "stop-loss hit", "take_profit": "take-profit hit"}.get(reason, reason)
        body = (f"<h2>{word}: {_e(symbol)} closed {pnl_eur:+.2f} EUR ({return_pct:+.2f}%)</h2>"
                f"<p>Entered at {entry:.4f}, exited at {exit_:.4f} ({_e(why)}).</p>"
                + (f"<p>Account is now at <b>€{equity:.2f}</b>.</p>" if equity is not None else "")
                + "<p style='color:#888'>Paper money only. The Historian is writing the lesson.</p>")
        return await self.send(f"{word}: {symbol} {pnl_eur:+.2f} EUR ({why})", body)

    async def breaker_tripped(self, until: str, equity: float) -> bool:
        body = (f"<h2>Circuit breaker tripped</h2><p>The account fell 5% from its 24h high "
                f"(now €{equity:.2f}). No new trades until {_e(until)}.</p>")
        return await self.send("Ultron paused trading for 24h", body)


_last_test = 0.0


def mask(address: str) -> str:
    name, _, domain = address.partition("@")
    return (name[:1] + "***@" + domain) if domain else ""


def send_test_email(db_path: str, env: dict | None = None) -> dict:
    """Used by the dashboard's test button. Synchronous, because it runs on the web thread.

    Returns {"ok": bool, "detail": str, "to": masked address}. Limited to one per 30 seconds.
    """
    global _last_test
    env = env if env is not None else os.environ
    key, to = env.get("RESEND_API_KEY", ""), env.get("EMAIL_TO", "")
    sender = env.get("EMAIL_FROM", "Ultron <onboarding@resend.dev>")
    if not key or not to:
        return {"ok": False, "detail": "RESEND_API_KEY or EMAIL_TO is not set in the bot's environment",
                "to": mask(to)}
    if time.time() - _last_test < 30:
        return {"ok": False, "detail": "wait 30 seconds between test emails", "to": mask(to)}
    _last_test = time.time()
    subject = "Ultron test email"
    try:
        r = httpx.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=20, json={
            "from": sender, "to": [to], "subject": subject,
            "html": "<h2>It works</h2><p>This came from the test button on your dashboard. "
                    "You'll get an email like this for every paper trade the desk opens or closes.</p>"})
        ok = r.status_code < 300
        detail = ("accepted by Resend. Check your inbox and spam folder." if ok
                  else f"HTTP {r.status_code} {r.text[:200]}")
    except Exception as e:
        ok, detail = False, str(e)[:200]
    try:
        db = sqlite3.connect(db_path, timeout=5)
        db.execute("CREATE TABLE IF NOT EXISTS emails (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
                   " subject TEXT NOT NULL, ok INTEGER NOT NULL, detail TEXT)")
        db.execute("INSERT INTO emails (ts, subject, ok, detail) VALUES (?,?,?,?)",
                   (datetime.now(timezone.utc).isoformat(), subject + " (button)", int(ok), detail[:300]))
        db.commit()
        db.close()
    except sqlite3.Error:
        pass
    return {"ok": ok, "detail": detail, "to": mask(to)}
