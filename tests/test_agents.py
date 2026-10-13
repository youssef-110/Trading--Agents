import asyncio
import dataclasses
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Scout, Setup, Ultron  # noqa: E402
from config import ScoutRules, Settings  # noqa: E402
from data_feed import add_indicators  # noqa: E402
from llm import CallBudget, LLMError, LLMRouter, ProviderDisabled, parse_json  # noqa: E402
from main import Bot  # noqa: E402
from memory_loop import LessonStore, Scoreboard, ShadowTracker  # noqa: E402
from portfolio import Portfolio  # noqa: E402
from config import ExecutionCosts  # noqa: E402

T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)   # a Thursday, US market closed at 12:00 UTC


class FakeProvider:
    def __init__(self, name, reply=None, error=None):
        self.name, self.reply, self.error, self.calls = name, reply, error, []

    async def complete_json(self, system, user, temperature=0.4):
        self.calls.append(system)
        if self.error:
            raise self.error
        return self.reply(system, user) if callable(self.reply) else dict(self.reply)


def router(db, providers, daily=100):
    return LLMRouter({p.name: p for p in providers}, CallBudget(db),
                     {p.name: (daily, 1000) for p in providers})


def run(coro):
    return asyncio.run(coro)


# ---------- llm plumbing ----------

def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"action": "HOLD"} hope that helps') == {"action": "HOLD"}


def test_budget_is_a_hard_stop_and_persists(tmp_path):
    db = Portfolio(str(tmp_path / "b.db"), 50, ExecutionCosts()).db
    p = FakeProvider("gemini", {"ok": True})
    r = router(db, [p], daily=2)
    assert run(r.ask("s", "u")) and run(r.ask("s", "u"))
    assert run(r.ask("s", "u")) is None
    assert len(p.calls) == 2
    r2 = router(db, [p], daily=2)          # restart: count survives
    assert run(r2.ask("s", "u")) is None


def test_dead_grok_falls_back_to_gemini(tmp_path):
    db = Portfolio(str(tmp_path / "f.db"), 50, ExecutionCosts()).db
    grok = FakeProvider("grok", error=ProviderDisabled("402 no credits"))
    gem = FakeProvider("gemini", {"x": 1})
    r = router(db, [grok, gem])
    assert run(r.ask("s", "u", prefer=("grok", "gemini")))[1] == "gemini"
    run(r.ask("s", "u", prefer=("grok", "gemini")))
    assert len(grok.calls) == 1            # switched off after the first failure


# ---------- the desk ----------

def uptrend_then_dip(n=260, dip=6):
    up = np.linspace(100, 200, n - dip)
    down = up[-1] * np.cumprod(np.full(dip, 0.985))
    close = pd.Series(np.concatenate([up, down]))
    idx = pd.date_range(T0 - timedelta(hours=n), periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": close.values, "high": close.values * 1.002,
                       "low": close.values * 0.998, "close": close.values,
                       "volume": 100.0}, index=idx)
    return df


def make_desk(tmp_path, providers):
    pf = Portfolio(str(tmp_path / "d.db"), 50, ExecutionCosts())
    scores = Scoreboard(pf.db)
    lessons, shadows = LessonStore(pf.db), ShadowTracker(pf.db, scores)
    return Ultron(router(pf.db, providers), lessons, scores, shadows), providers


def test_scout_flags_oversold_pullback():
    setup = Scout(ScoutRules()).scan("BTC/EUR", "crypto", add_indicators(uptrend_then_dip()))
    assert setup is not None and setup.setup_type == "oversold_pullback_in_uptrend"


def a_setup():
    return Scout(ScoutRules()).scan("BTC/EUR", "crypto", add_indicators(uptrend_then_dip()))


def test_weak_council_never_wakes_ultron(tmp_path):
    def reply(system, user):
        return {"action": "BUY" if "You are the bull" in system else "HOLD",
                "confidence": 0.9, "thesis": "t"}
    gem = FakeProvider("gemini", reply)
    ultron, _ = make_desk(tmp_path, [gem])
    v = run(ultron.deliberate(a_setup()))
    assert v.signal.action == "HOLD"
    assert len(gem.calls) == 9             # 6 specialists + 3 leads; no Ultron call
    assert not any("Ultron" in s for s in gem.calls)


def test_strong_council_and_ultron_buy(tmp_path):
    gem = FakeProvider("gemini", {"action": "BUY", "confidence": 0.8, "thesis": "bounce"})
    grok = FakeProvider("grok", {"action": "BUY", "confidence": 0.7, "thesis": "trend intact"})
    ultron, _ = make_desk(tmp_path, [gem, grok])
    v = run(ultron.deliberate(a_setup()))
    assert v.signal.action == "BUY" and v.signal.confidence == 0.8
    providers = {x.agent: x.provider for x in v.votes}
    assert providers["bull"] == providers["bull.momentum"] == "grok"
    assert [x.agent for x in v.votes] == [
        "analyst.trend", "analyst.reversion", "bull.momentum", "bull.pullback",
        "bear.trap", "bear.regime", "analyst", "bull", "bear", "ultron"]


def test_no_budget_means_hold(tmp_path):
    gem = FakeProvider("gemini", {"action": "BUY", "confidence": 0.9})
    pf = Portfolio(str(tmp_path / "n.db"), 50, ExecutionCosts())
    scores = Scoreboard(pf.db)
    u = Ultron(router(pf.db, [gem], daily=0), LessonStore(pf.db), scores, ShadowTracker(pf.db, scores))
    assert run(u.deliberate(a_setup())).signal.action == "HOLD"


def test_garbage_vote_is_ignored(tmp_path):
    gem = FakeProvider("gemini", {"action": "MOON", "confidence": "lots"})
    ultron, _ = make_desk(tmp_path, [gem])
    assert run(ultron.deliberate(a_setup())).signal.action == "HOLD"


def test_lead_sides_with_the_better_specialist(tmp_path):
    def reply(system, user):
        if "lead on a trading desk" in system:
            return {"winner": "bull.pullback", "action": "BUY", "confidence": 0.7, "thesis": "dip"}
        if "dip buy" in system:
            return {"action": "BUY", "confidence": 0.8, "thesis": "strong dip"}
        return {"action": "HOLD", "confidence": 0.3, "thesis": "meh"}
    gem = FakeProvider("gemini", reply)
    ultron, _ = make_desk(tmp_path, [gem])
    v = run(ultron.deliberate(a_setup()))
    bull = next(x for x in v.votes if x.agent == "bull")
    assert bull.winner == "bull.pullback"
    assert next(x for x in v.votes if x.agent == "analyst").winner in ("", "analyst.trend",
                                                                        "analyst.reversion")


def test_lead_failure_falls_back_to_cautious_reading(tmp_path):
    # Specialists answer, then the lead calls fail: one BUY + one HOLD must not become a BUY.
    def reply(system, user):
        if "lead on a trading desk" in system:
            raise LLMError("lead call failed")
        return {"action": "BUY" if "dip buy" in system else "HOLD", "confidence": 0.8, "thesis": "t"}
    gem = FakeProvider("gemini", reply)
    ultron, _ = make_desk(tmp_path, [gem])
    v = run(ultron.deliberate(a_setup()))
    assert v.signal.action == "HOLD"
    leads = {x.agent: x for x in v.votes if x.provider == "fallback"}
    assert set(leads) == {"analyst", "bull", "bear"} and all(x.action == "HOLD" for x in leads.values())


def test_specialists_are_graded_too(tmp_path):
    pf = Portfolio(str(tmp_path / "g.db"), 50, ExecutionCosts())
    scores = Scoreboard(pf.db)
    scores.record([{"agent": "bull.pullback", "action": "BUY"},
                   {"agent": "bear.trap", "action": "HOLD"}], "win")
    assert scores.reliability("bull.pullback") > 0.5 > scores.reliability("bear.trap")


# ---------- learning ----------

def test_shadow_grades_agents(tmp_path):
    pf = Portfolio(str(tmp_path / "s.db"), 50, ExecutionCosts())
    scores = Scoreboard(pf.db)
    sh = ShadowTracker(pf.db, scores)
    setup = Setup("X", "crypto", "macd_bull_cross_above_ema50", 100.0,
                  str(pd.Timestamp(T0)), {"close": 100, "ema200": 90, "atr_pct": 0.5})
    votes = [{"agent": "bull", "action": "BUY"}, {"agent": "bear", "action": "HOLD"}]
    sh.record(setup, "HOLD", False, votes, 2.0, 4.0, 48)
    idx = pd.date_range(T0 + timedelta(hours=1), periods=3, freq="1h", tz="UTC")
    later = pd.DataFrame({"open": 101.0, "high": [101.0, 103.0, 105.0], "low": 100.5,
                          "close": 102.0, "volume": 1.0}, index=idx)
    assert sh.resolve("X", later, T0 + timedelta(hours=4))[0]["outcome"] == "win"
    assert scores.reliability("bull") > 0.5 > scores.reliability("bear")
    summary = sh.summary()
    assert summary["ultron_passed"]["wins"] == 1  # a missed winner shows up


def test_lessons_match_the_situation(tmp_path):
    db = Portfolio(str(tmp_path / "l.db"), 50, ExecutionCosts()).db
    ls = LessonStore(db)
    ls.add("SOL/EUR", "crypto", "macd_bull_cross_above_ema50", "downtrend/volatile", "loss", -2.3,
           "macd crosses in downtrend fail", T0)
    ls.add("AAPL", "stock", "ema20_reclaim_on_volume", "uptrend/calm", "win", 3.6, "unrelated", T0)
    hits = ls.relevant("ETH/EUR", "crypto", "macd_bull_cross_above_ema50", "downtrend/volatile")
    assert [h["lesson"] for h in hits] == ["macd crosses in downtrend fail"]


# ---------- end to end ----------

class FakeData:
    def __init__(self, df):
        self.df = df

    async def bars_eur(self, symbol, asset_class):
        return add_indicators(self.df)


def test_full_cycle_trade_stop_and_lesson(tmp_path):
    settings = dataclasses.replace(Settings(), db_path=str(tmp_path / "e.db"),
                                   report_path=str(tmp_path / "r.json"),
                                   crypto_symbols=("BTC/EUR",), stock_symbols=())

    def reply(system, user):
        if "historian" in system:
            return {"lesson": "dip-buying into a fresh 6-bar selloff kept falling"}
        return {"action": "BUY", "confidence": 0.8, "thesis": "pullback in uptrend"}
    gem = FakeProvider("gemini", reply)

    # Add a forming bar after the dip: the bot must only judge the closed ones.
    df = uptrend_then_dip()
    forming = df.iloc[[-1]].copy()
    forming.index = forming.index + timedelta(hours=1)
    df = pd.concat([df, forming])

    bot = Bot(settings, FakeData(df), {"gemini": gem}, {"gemini": (100, 1000)})
    run(bot.cycle(T0))
    pos = bot.pf.open_positions()
    assert len(pos) == 1 and pos[0].cost_eur == 100.0     # 20% of EUR 500
    assert pos[0].meta["setup_type"] == "oversold_pullback_in_uptrend"

    calls_before = len(gem.calls)
    run(bot.cycle(T0 + timedelta(minutes=5)))     # same candle: no new deliberation
    assert len(gem.calls) == calls_before

    # Next candle crashes 4%: hard stop fires, historian writes a lesson.
    crash = df.iloc[[-1]].copy()
    crash.index = crash.index + timedelta(hours=1)
    crash[["open", "high", "low", "close"]] *= 0.96
    bot.data = FakeData(pd.concat([df, crash]))
    run(bot.cycle(T0 + timedelta(hours=1, minutes=5)))
    assert bot.pf.open_positions() == []
    t = bot.pf.trades()[0]
    assert t["exit_reason"] == "stop_loss"
    assert bot.lessons.count() == 1

    report = json.loads(Path(settings.report_path).read_text())
    assert report["performance"]["trades"] == 1
    assert report["llm_budget_today"]["gemini"]["used_today"] == len(gem.calls)


def test_dashboard_state_after_a_cycle(tmp_path):
    from dashboard import build_state
    settings = dataclasses.replace(Settings(), db_path=str(tmp_path / "w.db"),
                                   report_path=str(tmp_path / "r.json"),
                                   crypto_symbols=("BTC/EUR",), stock_symbols=())
    gem = FakeProvider("gemini", {"action": "BUY", "confidence": 0.8, "thesis": "x"})
    df = uptrend_then_dip()
    forming = df.iloc[[-1]].copy()
    forming.index = forming.index + timedelta(hours=1)
    bot = Bot(settings, FakeData(pd.concat([df, forming])), {"gemini": gem}, {"gemini": (100, 1000)})
    run(bot.cycle(T0))
    st = build_state(settings.db_path, {"gemini": (100, 1000)})
    assert st["cycles"][0]["loaded"] == "BTC/EUR" and st["cycles"][0]["setups"] == 1
    assert len(st["debates"]) == 1 and len(st["debates"][0]["votes"]) == 10
    assert [l["name"] for l in st["tree"]["leads"]] == ["analyst", "bull", "bear"]
    assert all(len(l["subs"]) == 2 for l in st["tree"]["leads"])
    assert st["llm"]["gemini"]["used"] == 10 and len(st["positions"]) == 1
    json.dumps(st, default=str)


def test_dashboard_never_goes_public_without_a_token():
    from dashboard import bind_settings
    assert bind_settings({}) == ("127.0.0.1", 8000, "")                 # local and open is fine
    assert bind_settings({"PORT": "10000"}) is None                     # public, no token: refuse
    assert bind_settings({"PORT": "10000", "DASHBOARD_TOKEN": "s"}) == ("0.0.0.0", 10000, "s")


# ---------- context, budget guard, email ----------

def test_setup_carries_derived_numbers():
    ctx = a_setup().context
    assert ctx["stop_distance_in_atr"] > 0 and ctx["target_distance_in_atr"] == 2 * ctx["stop_distance_in_atr"]
    assert ctx["ret_6h"] < 0 and ctx["dist_ema20_pct"] < 0     # the fixture is a 6-bar dip
    assert 0 <= ctx["range_pos_24h"] <= 1 and ctx["weekday"]


def test_vote_keeps_the_evidence_it_cites():
    from agents import parse_vote
    v = parse_vote("analyst", {"action": "BUY", "confidence": 0.7,
                               "evidence": ["rsi14=29.1", "dist_ema200_pct=+1.8", "a", "dropped"]}, "gemini")
    assert v.evidence == ["rsi14=29.1", "dist_ema200_pct=+1.8", "a"]
    assert parse_vote("x", {"action": "BUY", "confidence": 0.7, "evidence": "oops"}, "g").evidence == []


def test_low_budget_skips_the_debate_and_retries_the_candle(tmp_path):
    settings = dataclasses.replace(Settings(), db_path=str(tmp_path / "q.db"),
                                   report_path=str(tmp_path / "r.json"),
                                   crypto_symbols=("BTC/EUR",), stock_symbols=())
    gem = FakeProvider("gemini", {"action": "BUY", "confidence": 0.8, "thesis": "x"})
    df = uptrend_then_dip()
    forming = df.iloc[[-1]].copy()
    forming.index = forming.index + timedelta(hours=1)
    bot = Bot(settings, FakeData(pd.concat([df, forming])), {"gemini": gem}, {"gemini": (9, 1000)})
    run(bot.cycle(T0))
    assert gem.calls == [] and bot.shadows.summary()["pending"] == 0     # nothing half-graded
    assert bot.pf.get_state("last_bar:BTC/EUR") is None                   # will look again


class FakeResend:
    def __init__(self, status=200):
        self.status, self.sent = status, []

    def client(self):
        import httpx

        def handler(request):
            self.sent.append((dict(request.headers), json.loads(request.content)))
            return httpx.Response(self.status, json={"id": "x"})
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_email_goes_out_on_open_and_close_and_never_raises():
    from notify import Notifier

    async def go(status):
        fake = FakeResend(status)
        async with fake.client() as http:
            n = Notifier(http, api_key="k", to="me@example.com")
            a = await n.trade_opened("BTC/EUR", 10.0, 100.0, 98.0, 104.0, "dip <b>", [])
            b = await n.trade_closed("BTC/EUR", -0.22, -2.0, "stop_loss", 100.0, 98.0, 49.8)
            return fake, a, b
    fake, a, b = run(go(200))
    assert a and b and len(fake.sent) == 2
    assert fake.sent[0][0]["authorization"] == "Bearer k" and fake.sent[0][1]["to"] == ["me@example.com"]
    assert "Loss" in fake.sent[1][1]["subject"] and "&lt;b&gt;" in fake.sent[0][1]["html"]
    _, a, b = run(go(500))
    assert not a and not b                                    # a Resend outage is just a warning
    assert not run(Notifier(None).send("s", "b"))              # no key, no http: quietly off


# ---------- backup and health ----------

class FakeGitHub:
    """Just enough of the GitHub contents API for one file on one branch."""
    def __init__(self):
        self.file, self.sha, self.puts, self.branch_made = None, None, 0, False

    def client(self):
        import base64, httpx

        def handler(req):
            path = req.url.path
            if req.method == "GET" and "/git/ref/heads/state" in path:
                return httpx.Response(200 if self.branch_made else 404, json={"object": {"sha": "b"}})
            if req.method == "GET" and "/git/ref/heads/main" in path:
                return httpx.Response(200, json={"object": {"sha": "m"}})
            if req.method == "GET" and path.endswith("/repos/me/r"):
                return httpx.Response(200, json={"default_branch": "main"})
            if req.method == "POST" and path.endswith("/git/refs"):
                self.branch_made = True
                return httpx.Response(201, json={})
            if path.endswith("/contents/ledger.db.gz"):
                if req.method == "GET":
                    if self.file is None:
                        return httpx.Response(404, json={})
                    return httpx.Response(200, json={"content": base64.b64encode(self.file).decode(), "sha": self.sha})
                body = json.loads(req.content)
                assert body["branch"] == "state" and (self.file is None or body["sha"] == self.sha)
                self.file, self.sha, self.puts = base64.b64decode(body["content"]), f"s{self.puts}", self.puts + 1
                return httpx.Response(200, json={})
            return httpx.Response(500, json={"path": path})
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_backup_round_trip_restores_the_ledger(tmp_path):
    from backup import Backup
    gh = FakeGitHub()

    async def go():
        async with gh.client() as http:
            pf = Portfolio(str(tmp_path / "a.db"), 50, ExecutionCosts())
            LessonStore(pf.db).add("X", "crypto", "s", "r", "loss", -2.0, "keep me", T0)
            b = Backup(http, str(tmp_path / "a.db"), token="t", repo="me/r", min_gap_s=3600)
            assert await b.maybe_backup() and not await b.maybe_backup()       # second one is throttled
            assert await b.maybe_backup(force=True) and gh.puts == 2          # sha handling on update
            fresh = Backup(http, str(tmp_path / "gone.db"), token="t", repo="me/r")
            assert await fresh.restore_if_missing()
            return Portfolio(str(tmp_path / "gone.db"), 99, ExecutionCosts())
    restored = run(go())
    assert LessonStore(restored.db).count() == 1
    assert restored.starting_balance == 99      # it never traded, so a new balance in config still applies


def test_backup_is_off_without_a_token_and_never_raises(tmp_path):
    from backup import Backup
    assert not run(Backup(None, str(tmp_path / "x.db")).maybe_backup())

    async def broken():
        import httpx
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as http:
            Portfolio(str(tmp_path / "y.db"), 50, ExecutionCosts())
            return await Backup(http, str(tmp_path / "y.db"), token="t", repo="me/r").maybe_backup()
    assert run(broken()) is False


def test_health_goes_red_when_the_loop_goes_quiet(tmp_path):
    from dashboard import health
    from memory_loop import ActivityLog
    pf = Portfolio(str(tmp_path / "h.db"), 50, ExecutionCosts())
    assert health(str(tmp_path / "h.db"))[0]                       # just started: fine
    log_ = ActivityLog(pf.db)
    log_.cycle(datetime.now(timezone.utc), [], [], False, 0, False, 50.0)
    assert health(str(tmp_path / "h.db"))[0]
    log_.cycle(datetime.now(timezone.utc) - timedelta(minutes=45), [], [], False, 0, False, 50.0)
    assert not health(str(tmp_path / "h.db"))[0]


def test_a_busy_provider_gets_one_retry(tmp_path):
    from llm import Overloaded
    db = Portfolio(str(tmp_path / "o.db"), 50, ExecutionCosts()).db
    state = {"n": 0}

    def reply(system, user):
        state["n"] += 1
        if state["n"] == 1:
            raise Overloaded("gemini: HTTP 503")
        return {"ok": True}
    gem = FakeProvider("gemini", reply)
    r = router(db, [gem])
    r.retry_wait_s = 0
    assert run(r.ask("s", "u"))[0] == {"ok": True} and len(gem.calls) == 2
    assert r.budget.used("gemini") == 2                    # the retry is counted against the cap
    state["n"] = -99                                       # still failing after the retry: give up
    gem.reply = lambda s, u: (_ for _ in ()).throw(Overloaded("again"))
    assert run(r.ask("s", "u")) is None


def test_cerebras_provider_speaks_openai_format_and_bear_prefers_it(tmp_path):
    import httpx
    from llm import CerebrasProvider, build_providers
    seen = {}

    def handler(req):
        seen["url"], seen["auth"] = str(req.url), req.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"action": "HOLD"}'}}]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            p = CerebrasProvider("csk-test", "some-model", http)
            return await p.complete_json("s", "u")
    assert run(go()) == {"action": "HOLD"}
    assert seen["url"] == "https://api.cerebras.ai/v1/chat/completions" and seen["auth"] == "Bearer csk-test"
    cfg = dataclasses.replace(Settings().llm, gemini_api_key=None, xai_api_key=None, cerebras_api_key="csk-x")
    assert list(build_providers(cfg, None)[0]) == ["cerebras"]

    gem = FakeProvider("gemini", {"action": "HOLD", "confidence": 0.5, "thesis": "t"})
    cer = FakeProvider("cerebras", {"action": "HOLD", "confidence": 0.5, "thesis": "t"})
    ultron, _ = make_desk(tmp_path, [gem, cer])
    run(ultron.deliberate(a_setup()))
    assert any("You are the bear" in s for s in cer.calls) and not any("You are the bear" in s for s in gem.calls)


def test_email_test_button_logs_result_and_is_rate_limited(tmp_path, monkeypatch):
    import httpx
    import notify
    from dashboard import build_state
    Portfolio(str(tmp_path / "t.db"), 50, ExecutionCosts())
    calls = []
    monkeypatch.setattr(notify.httpx, "post",
                        lambda url, **kw: calls.append(kw["json"]) or httpx.Response(200, json={"id": "x"}))
    monkeypatch.setattr(notify, "_last_test", 0.0)
    env = {"RESEND_API_KEY": "k", "EMAIL_TO": "me@example.com"}
    out = notify.send_test_email(str(tmp_path / "t.db"), env)
    assert out["ok"] and out["to"] == "m***@example.com" and calls[0]["to"] == ["me@example.com"]
    assert not notify.send_test_email(str(tmp_path / "t.db"), env)["ok"]          # too soon
    st = build_state(str(tmp_path / "t.db"))
    assert st["email"]["recent"][0]["ok"] == 1 and "scout" in st
    monkeypatch.setattr(notify, "_last_test", 0.0)
    assert "not set" in notify.send_test_email(str(tmp_path / "t.db"), {})["detail"]


def test_cycle_records_how_many_candles_the_scout_checked(tmp_path):
    from dashboard import build_state
    settings = dataclasses.replace(Settings(), db_path=str(tmp_path / "c.db"),
                                   report_path=str(tmp_path / "r.json"),
                                   crypto_symbols=("BTC/EUR",), stock_symbols=())
    gem = FakeProvider("gemini", {"action": "HOLD", "confidence": 0.5, "thesis": "t"})
    df = uptrend_then_dip()
    forming = df.iloc[[-1]].copy()
    forming.index = forming.index + timedelta(hours=1)
    bot = Bot(settings, FakeData(pd.concat([df, forming])), {"gemini": gem}, {"gemini": (100, 1000)})
    run(bot.cycle(datetime.now(timezone.utc)))
    run(bot.cycle(datetime.now(timezone.utc)))             # same candle again: nothing new to check
    st = build_state(settings.db_path)
    assert [c["checked"] for c in st["cycles"]] == [0, 1]
    assert st["scout"]["candles_checked_24h"] == 1 and st["scout"]["setups_flagged_24h"] == 1


def test_wrong_key_gets_a_login_box_and_special_characters_work(tmp_path, monkeypatch):
    import socket
    import urllib.error
    import urllib.parse
    import urllib.request
    import dashboard
    Portfolio(str(tmp_path / "l.db"), 50, ExecutionCosts())
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    secret = "p@ss w&rd#1+x"
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.setenv("DASHBOARD_PORT", str(port))
    monkeypatch.setenv("DASHBOARD_TOKEN", secret)
    server = dashboard.serve(str(tmp_path / "l.db"))
    base = f"http://127.0.0.1:{port}"

    def get(path):
        try:
            with urllib.request.urlopen(base + path, timeout=10) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
    try:
        code, body = get("/")
        assert code == 401 and "<form" in body and "Dashboard key" in body       # a login box, not bare text
        assert get("/ultron?key=nope")[0] == 401 and get("/api/state")[0] == 401
        assert get("/health")[0] == 200
        q = urllib.parse.quote(secret, safe="")                                   # what the login box sends
        assert get(f"/?key={q}")[0] == 200 and get(f"/api/state?key={q}")[0] == 200
    finally:
        server.shutdown()


def _bull_only_bot(tmp_path, practice_on=True):
    from config import PracticeMode
    settings = dataclasses.replace(Settings(), db_path=str(tmp_path / "p.db"),
                                   report_path=str(tmp_path / "r.json"),
                                   crypto_symbols=("BTC/EUR",), stock_symbols=(),
                                   practice=PracticeMode(enabled=practice_on))

    def reply(system, user):
        if "You are the bull on" in system:
            return {"action": "BUY", "confidence": 0.7, "thesis": "dip should hold"}
        return {"action": "HOLD", "confidence": 0.6, "thesis": "not convinced"}
    gem = FakeProvider("gemini", reply)
    df = uptrend_then_dip()
    forming = df.iloc[[-1]].copy()
    forming.index = forming.index + timedelta(hours=1)
    return Bot(settings, FakeData(pd.concat([df, forming])), {"gemini": gem}, {"gemini": (100, 1000)})


def test_lone_bull_vote_becomes_a_small_practice_trade(tmp_path):
    bot = _bull_only_bot(tmp_path)
    run(bot.cycle(T0))
    pos = bot.pf.open_positions()
    assert len(pos) == 1 and pos[0].cost_eur == 25.0 and pos[0].meta["practice"] is True
    assert pos[0].thesis.startswith("[practice]")
    # The desk's own verdict stays HOLD, so "Ultron said BUY" stats aren't polluted.
    row = bot.pf.db.execute("SELECT final_action, executed FROM shadow_setups").fetchone()
    assert (row["final_action"], row["executed"]) == ("HOLD", 1)


def test_no_practice_trade_when_the_mode_is_off(tmp_path):
    bot = _bull_only_bot(tmp_path, practice_on=False)
    run(bot.cycle(T0))
    assert bot.pf.open_positions() == []


def test_cycle_writes_the_live_board_for_the_dashboard(tmp_path):
    from dashboard import build_state
    bot = _bull_only_bot(tmp_path)
    run(bot.cycle(T0))
    run(bot.refresh_live_prices())                      # FakeData has no live feed: must not raise
    st = build_state(bot.s.db_path, {"gemini": (100, 1000)})
    coin = st["watchlist"]["coins"]["BTC/EUR"]
    assert coin["rsi"] < 35 and len(coin["spark"]) == 48 and coin["held"] is True
    assert st["positions"][0]["practice"] is True and set(st["funnel"]) == {"checked", "flagged", "debated", "opened"}
    assert st["by_setup"] == {} and st["cycle_seconds"] == 300


def test_quiet_hour_desk_pick_debates_the_best_coin(tmp_path):
    from agents import Scout as _S
    flat = uptrend_then_dip(dip=0)             # steady uptrend: no Scout rule fires
    assert _S(ScoutRules()).scan("BTC/EUR", "crypto", add_indicators(flat)) is None
    settings = dataclasses.replace(Settings(), db_path=str(tmp_path / "q.db"),
                                   report_path=str(tmp_path / "r.json"),
                                   crypto_symbols=("BTC/EUR",), stock_symbols=())
    gem = FakeProvider("gemini", {"action": "HOLD", "confidence": 0.6, "p_win": 0.3, "thesis": "meh"})
    forming = flat.iloc[[-1]].copy()
    forming.index = forming.index + timedelta(hours=1)
    bot = Bot(settings, FakeData(pd.concat([flat, forming])), {"gemini": gem}, {"gemini": (100, 1000)})
    run(bot.cycle(T0))
    row = bot.pf.db.execute("SELECT setup_type FROM debates").fetchone()
    assert row["setup_type"] == "desk_pick"
    run(bot.cycle(T0 + timedelta(minutes=30)))  # same quiet stretch: no second pick yet
    assert bot.pf.db.execute("SELECT COUNT(*) FROM debates").fetchone()[0] == 1
    votes = json.loads(bot.pf.db.execute("SELECT votes FROM debates").fetchone()["votes"])
    assert votes[0]["p_win"] == 0.3


def test_a_rate_limited_provider_waits_and_retries_once(tmp_path):
    from llm import RateLimited
    db = Portfolio(str(tmp_path / "rl.db"), 50, ExecutionCosts()).db
    state = {"n": 0}

    def reply(system, user):
        state["n"] += 1
        if state["n"] == 1:
            raise RateLimited("gemini: rate limited")
        return {"ok": True}
    gem = FakeProvider("gemini", reply)
    r = router(db, [gem])
    r.rate_wait_s = 0
    assert run(r.ask("s", "u"))[0] == {"ok": True} and r.budget.used("gemini") == 2


def test_gemini_falls_through_to_the_next_model_when_one_is_stuck():
    import httpx as _httpx
    from llm import GeminiProvider
    seen = []

    def handler(request):
        model = request.url.path.split("/")[-1].split(":")[0]
        seen.append(model)
        if model == "slow-model":
            raise _httpx.ReadTimeout("too slow")
        if model == "gone-model":
            return _httpx.Response(404, json={"error": {"message": "no longer available"}})
        return _httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"action": "BUY"}'}]}}]})

    async def go():
        async with _httpx.AsyncClient(transport=_httpx.MockTransport(handler)) as http:
            g = GeminiProvider("k", "slow-model", http, ("gone-model", "good-model"))
            return await g.complete_json("s", "u")
    assert run(go()) == {"action": "BUY"} and seen == ["slow-model", "gone-model", "good-model"]


def test_a_debate_with_no_answers_says_so(tmp_path):
    from llm import LLMError
    u, _ = make_desk(tmp_path, [FakeProvider("gemini", error=LLMError("boom"))])
    setup = Scout(ScoutRules()).scan("BTC/EUR", "crypto", add_indicators(uptrend_then_dip()))
    v = run(u.deliberate(setup))
    assert v.signal.action == "HOLD" and "no agent answered" in v.reason


# ---------- staying alive on Render (2026-10-13 restart loop) ----------

def _bot_with_dips(tmp_path, symbols, provider, **over):
    settings = dataclasses.replace(Settings(), db_path=str(tmp_path / "k.db"),
                                   report_path=str(tmp_path / "r.json"),
                                   crypto_symbols=symbols, stock_symbols=(), **over)
    df = uptrend_then_dip()
    forming = df.iloc[[-1]].copy()
    forming.index = forming.index + timedelta(hours=1)
    return Bot(settings, FakeData(pd.concat([df, forming])), {"gemini": provider}, {"gemini": (500, 1000)})


def test_health_stays_green_while_a_long_debate_is_running(tmp_path):
    from dashboard import health
    from memory_loop import ActivityLog
    pf = Portfolio(str(tmp_path / "h.db"), 50, ExecutionCosts())
    ActivityLog(pf.db).cycle(datetime.now(timezone.utc) - timedelta(minutes=45), [], [], False, 0, False, 50.0)
    assert not health(str(tmp_path / "h.db"))[0]
    pf.set_state("heartbeat", datetime.now(timezone.utc).isoformat())   # loop is busy, not dead
    assert health(str(tmp_path / "h.db"))[0]


def test_one_cycle_debates_only_a_few_setups_and_keeps_the_rest_for_later(tmp_path):
    gem = FakeProvider("gemini", {"action": "HOLD", "confidence": 0.3, "thesis": "x"})
    bot = _bot_with_dips(tmp_path, ("BTC/EUR", "ETH/EUR", "SOL/EUR"), gem, max_debates_per_cycle=2)
    run(bot.cycle(T0))
    seen = [s for s in ("BTC/EUR", "ETH/EUR", "SOL/EUR") if bot.pf.get_state(f"last_bar:{s}")]
    assert len(seen) == 2                                    # third candle is left for the next cycle
    run(bot.cycle(T0 + timedelta(minutes=5)))
    assert all(bot.pf.get_state(f"last_bar:{s}") for s in ("BTC/EUR", "ETH/EUR", "SOL/EUR"))


def test_a_debate_that_runs_too_long_is_dropped_and_the_cycle_finishes(tmp_path):
    class Slow(FakeProvider):
        async def complete_json(self, system, user, temperature=0.4):
            await asyncio.sleep(5)
            return {"action": "BUY", "confidence": 0.9, "thesis": "late"}
    bot = _bot_with_dips(tmp_path, ("BTC/EUR",), Slow("gemini"), debate_timeout_s=0.2)
    run(bot.cycle(T0))
    assert bot.pf.open_positions() == []
    assert bot.pf.db.execute("SELECT COUNT(*) FROM cycles").fetchone()[0] == 1


def test_health_says_whether_the_backup_works(tmp_path):
    import backup
    from dashboard import health
    Portfolio(str(tmp_path / "h.db"), 50, ExecutionCosts())
    backup.Backup(None, str(tmp_path / "h.db"), token="", repo="me/r")
    assert "backup off" in health(str(tmp_path / "h.db"))[1]
    assert "BACKUP_GITHUB_TOKEN" in health(str(tmp_path / "h.db"))[1]
