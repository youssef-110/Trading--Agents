"""The trading desk.

  Scout      deterministic. Scans closed candles and flags setups. Zero API calls.
  Analyst    asset-class lead (crypto or stocks). Judges its two specialists.
  Bull       argues the long case. Prefers Grok, so the debate isn't one model
             arguing with itself. Judges its two specialists.
  Bear       reads the bull's argument and attacks it. Hunts for traps. Judges
             its two specialists.
  Specialists  two per lead (see SUBS). They answer independently, then the lead
             picks the better-argued answer and names the winner.
  Ultron     the manager. Runs the debate, weighs agents by track record, reads
             past lessons and the desk's history with this setup, makes the call.
  Historian  writes a post-mortem lesson for every closed trade.
  Risk       lives in risk_manager.py. It has the final veto and Ultron can't
             override it.

Every agent fails safe: no budget, bad JSON or an API error means HOLD.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime

import pandas as pd

from config import RiskLimits, ScoutRules
from data_feed import snapshot
from llm import LLMRouter
from memory_loop import LessonStore, Scoreboard, ShadowTracker, regime
from portfolio import ClosedTrade
from risk_manager import Signal

log = logging.getLogger("desk")

COUNCIL_MIN_SCORE = 0.40       # weighted BUY strength needed before Ultron is even asked
COUNCIL = ("analyst", "bull", "bear")
DEBATE_CALLS = 10              # 6 specialists + 3 leads + Ultron, at most

# Two specialists under each lead. Different lenses on purpose, so they can disagree.
SUBS = {
    "analyst": (
        ("analyst.trend", "Your lens is trend following. Check EMA20/50/200 stacking and "
                          "dist_ema*_pct, the MACD histogram direction, and whether ret_24h agrees "
                          "with the setup. A BUY needs the higher trend on your side."),
        ("analyst.reversion", "Your lens is mean reversion. Check how stretched price is "
                              "(dist_ema20_pct, range_pos_24h), RSI extremes and rsi_change_3, and "
                              "whether selling looks exhausted (shrinking volume_ratio) or still "
                              "accelerating. Never catch a falling knife."),
    ),
    "bull": (
        ("bull.momentum", "Your angle is momentum: a strengthening MACD histogram, volume expansion "
                          "(volume_ratio), a reclaim of the EMAs, positive ret_6h. Say what is "
                          "confirming the move right now."),
        ("bull.pullback", "Your angle is the dip buy: a pullback inside an intact uptrend, oversold "
                          "RSI, support near the EMAs, price low in range_pos_24h. Say why this dip "
                          "should hold."),
    ),
    "bear": (
        ("bear.trap", "Your angle is traps: false breakouts, thin or fading volume, bull-trap "
                      "candles, momentum already rolling over (falling rsi_change_3, shrinking MACD "
                      "histogram), price already extended after a big ret_6h."),
        ("bear.regime", "Your angle is the bigger picture: the higher-timeframe trend, the stop "
                        "versus volatility (stop_distance_in_atr), the hour and weekday "
                        "(hour_utc, weekday), and whether desk_history says this setup fails here."),
    ),
}
SUB_NAMES = {lead: tuple(n for n, _ in subs) for lead, subs in SUBS.items()}

DESK_RULES = (
    "Context: long-only paper trading on 1-hour candles. Every trade exits automatically at "
    "-2% (stop) or +4% (take-profit); nothing else can happen. The only question is whether +4% "
    "arrives before -2% from here. Because the win is twice the loss, a trade only needs to win "
    "about 38% of the time to break even after fees. So this is not about certainty: estimate "
    "p_win, the chance that +4% comes first, and vote BUY when p_win is 0.42 or more. Vote HOLD "
    "when it is lower. Both are normal answers; the desk is graded on calibration, and an agent "
    "that never commits teaches it nothing.\n"
    "Method, in this order: (1) Trend: is price above the EMA200, and are EMA20/50 stacked with it? "
    "(2) Momentum and volume: do RSI, MACD histogram and volume_ratio confirm the move, or is it "
    "fading? (3) Room: stop_distance_in_atr and target_distance_in_atr show how the 2% / 4% levels "
    "compare with normal hourly noise. A stop under about 2 ATR gets hit by noise; a target over "
    "about 5 ATR rarely arrives within 48h. (4) History: lessons and desk_history show how this "
    "setup has actually done here, so weigh them over your instincts. (5) Estimate p_win and decide.\n"
    "Rules of evidence: use only the numbers you are given, and quote them exactly (for example "
    "'rsi14=29.1', 'dist_ema200_pct=+1.8'). Do not invent news, catalysts or price levels.\n"
    "Confidence is how strongly you hold your vote: 0.5 is marginal, 0.65 is a good case, 0.8 or "
    "more should be rare."
)
VOTE_FORMAT = ('Reply with JSON only: {"p_win": number 0-1, "action": "BUY" or "HOLD", '
               '"confidence": number 0-1, '
               '"evidence": ["2-3 short quotes of exact numbers from the data that decided it"], '
               '"thesis": "one or two sentences", "key_risk": "the main way this goes wrong"}')

ANALYST_PROMPTS = {
    "crypto": ("You are the crypto specialist on a trading desk. Crypto trades 24/7; weekend and "
               "late-night UTC volume is thin and breakouts on thin volume often fail. Coins move "
               "together with BTC, so a setup in a falling market is weaker. Judge trend alignment "
               "(EMA 20/50/200), momentum (RSI, MACD) and volume versus average."),
    "stock": ("You are the US equities specialist on a trading desk. The first and last hour of "
              "the session are noisy and overnight gaps can jump straight past a stop. Judge trend "
              "alignment (EMA 20/50/200), momentum (RSI, MACD) and volume versus average."),
}
BULL_PROMPT = ("You are the bull on a trading desk. Build the strongest HONEST case for taking this "
               "long, using only the data. A strong bull case names what is confirming (trend, "
               "momentum, volume) and why the 4% is reachable before the 2% stop. Remember the "
               "bar is a 42% chance, not a sure thing. If even the best honest case is below that, "
               "vote HOLD.")
BEAR_PROMPT = ("You are the bear on a trading desk. Your job is to find what is wrong with this "
               "trade: false breakouts, fading momentum, thin volume, price stretched far from the "
               "EMAs, fighting the higher trend, a stop that sits inside normal noise. You have read "
               "the bull's argument; attack its weakest claim with a number. Vote BUY only if you "
               "genuinely cannot find a serious problem, and say so.")
ULTRON_PROMPT = (
    "You are Ultron, the manager of this trading desk. You get the leads' votes, the six "
    "specialists' answers behind them, every agent's graded track record, the desk's real history "
    "with this exact setup, and lessons from past trades. Check three things before you decide: "
    "(a) did the leads agree for good reasons, or are they repeating each other's numbers? "
    "(b) does any lesson or the setup's history contradict the case? A history win rate under "
    "the 38% break-even needs a strong reason to override. (c) does the bear have a concrete "
    "number-backed objection nobody answered? Vote BUY if you believe the chance of +4% before "
    "-2% is at least 42%, a small margin over break-even for the models' habit of overrating "
    "themselves. You are graded on being right, and passing on good odds counts as a miss too.")
LEAD_PROMPT = (
    "You are the {lead} lead on a trading desk. Two of your specialists answered the same brief "
    "independently. Don't average them. Check each answer's claims against the numbers in the brief: "
    "an answer that quotes a number that isn't there, or reads one wrongly, loses. Prefer the answer "
    "with the better evidence over the more confident one (their track records are included, but a "
    "strong argument can beat a weak record). You may side with one, correct it, or vote HOLD if "
    "neither convinces you. Your answer becomes the {lead}'s vote.")
LEAD_FORMAT = ('Reply with JSON only: {"winner": "name of the specialist you sided with, or none", '
               '"p_win": number 0-1, "action": "BUY" or "HOLD", "confidence": number 0-1, '
               '"evidence": ["2-3 exact numbers that decided it"], '
               '"thesis": "one or two sentences", "key_risk": "the main way this goes wrong"}')
HISTORIAN_PROMPT = (
    "You are the desk historian. A trade just closed. Compare the entry thesis and the votes with "
    "what actually happened and write ONE concrete, checkable lesson for next time. Not a "
    "platitude like 'be careful'. Something like: 'MACD crosses on SOL in downtrend/volatile "
    "failed; require close above EMA200 first.' A win can still teach something. "
    'Reply with JSON only: {"lesson": "...", "verdict": "good_trade" | "bad_trade" | '
    '"good_trade_bad_luck"}')


@dataclass
class Setup:
    symbol: str
    asset_class: str
    setup_type: str
    price_eur: float
    bar_time: str
    indicators: dict
    context: dict = field(default_factory=dict)    # derived numbers so the models don't do arithmetic

    @property
    def regime(self) -> str:
        return regime(self.indicators)


@dataclass
class Vote:
    agent: str
    action: str
    confidence: float
    thesis: str
    key_risk: str
    provider: str
    winner: str = ""        # leads only: which specialist's answer they sided with
    evidence: list = field(default_factory=list)   # exact numbers the agent says decided it
    p_win: float | None = None                     # the agent's own estimate of +4% before -2%


@dataclass
class Verdict:
    signal: Signal
    votes: list[Vote]
    council_score: float
    reason: str
    lesson_ids: list[int] = field(default_factory=list)


def _evidence(raw) -> list:
    if not isinstance(raw, list):
        return []
    return [str(x)[:140] for x in raw[:3]]


def parse_vote(agent: str, raw: dict, provider: str) -> Vote | None:
    try:
        action = str(raw.get("action", "HOLD")).strip().upper()
        if action not in ("BUY", "HOLD"):
            action = "HOLD"
        conf = min(max(float(raw.get("confidence", 0.0)), 0.0), 1.0)
        try:
            p_win = min(max(float(raw["p_win"]), 0.0), 1.0)
        except (KeyError, TypeError, ValueError):
            p_win = None
        return Vote(agent, action, conf, str(raw.get("thesis", ""))[:600],
                    str(raw.get("key_risk", ""))[:300], provider,
                    str(raw.get("winner", "")).strip(), _evidence(raw.get("evidence")), p_win)
    except (TypeError, ValueError):
        log.warning("%s returned an unusable vote: %r", agent, raw)
        return None


# ---------------------------------------------------------------------------


def derive_context(df: pd.DataFrame) -> dict:
    """Numbers the agents would otherwise have to work out (badly) in their heads."""
    risk = RiskLimits()
    cur = df.iloc[-1]
    close = float(cur["close"])

    def ret(n: int):
        return round((close / float(df["close"].iloc[-1 - n]) - 1) * 100, 2) if len(df) > n else None

    def dist(col: str):
        return round((close / float(cur[col]) - 1) * 100, 2) if not pd.isna(cur[col]) else None

    day = df.iloc[-24:]
    lo, hi = float(day["low"].min()), float(day["high"].max())
    atr_pct = float(cur["atr_pct"])
    ts = df.index[-1]
    rsi_3 = df["rsi14"].iloc[-4] if len(df) > 4 else float("nan")
    return {
        "dist_ema20_pct": dist("ema20"), "dist_ema50_pct": dist("ema50"),
        "dist_ema200_pct": dist("ema200"),
        "ret_1h": ret(1), "ret_6h": ret(6), "ret_24h": ret(24),
        "rsi_change_3": None if pd.isna(rsi_3) else round(float(cur["rsi14"] - rsi_3), 1),
        "volume_ratio_last3": [None if pd.isna(v) else round(float(v), 2)
                               for v in df["volume_ratio"].iloc[-3:]],
        "range_pos_24h": round((close - lo) / (hi - lo), 2) if hi > lo else None,
        "stop_distance_in_atr": round(risk.stop_loss_pct / atr_pct, 1) if atr_pct else None,
        "target_distance_in_atr": round(risk.take_profit_pct / atr_pct, 1) if atr_pct else None,
        "hour_utc": ts.hour, "weekday": ts.day_name(),
    }


class Scout:
    def __init__(self, rules: ScoutRules):
        self.rules = rules

    def scan(self, symbol: str, asset_class: str, df: pd.DataFrame) -> Setup | None:
        """df must hold CLOSED bars only, with indicators already added."""
        if len(df) < 200:
            return None
        cur, prev = df.iloc[-1], df.iloc[-2]
        needed = ("close", "ema20", "ema50", "ema200", "rsi14", "macd_hist", "atr_pct")
        if any(pd.isna(cur[k]) for k in needed) or pd.isna(prev["macd_hist"]):
            return None
        if cur["atr_pct"] > self.rules.max_atr_pct:
            return None

        setup_type = None
        if cur["rsi14"] < self.rules.rsi_oversold and cur["close"] > cur["ema200"]:
            setup_type = "oversold_pullback_in_uptrend"
        elif (prev["close"] <= prev["ema20"] and cur["close"] > cur["ema20"]
              and cur["ema20"] > cur["ema50"]
              and (cur["volume_ratio"] or 0) >= self.rules.breakout_volume_ratio):
            setup_type = "ema20_reclaim_on_volume"
        elif prev["macd_hist"] <= 0 < cur["macd_hist"] and cur["close"] > cur["ema50"]:
            setup_type = "macd_bull_cross_above_ema50"

        if setup_type is None:
            return None
        return Setup(symbol, asset_class, setup_type, float(cur["close"]),
                     str(df.index[-1]), snapshot(df), derive_context(df))


    def best_candidate(self, frames: dict) -> Setup | None:
        """The coin that looks most like a long right now, for quiet hours. frames: sym -> (asset_class, closed df)."""
        best = None
        for sym, (ac, df) in frames.items():
            if len(df) < 200:
                continue
            cur, prev = df.iloc[-1], df.iloc[-2]
            if any(pd.isna(cur[k]) for k in ("ema20", "ema50", "ema200", "rsi14", "macd_hist", "atr_pct")):
                continue
            if cur["atr_pct"] > self.rules.max_atr_pct:
                continue
            score = (2 * (cur["close"] > cur["ema200"]) + (cur["ema20"] > cur["ema50"])
                     + (cur["macd_hist"] > prev["macd_hist"]) + (35 <= cur["rsi14"] <= 55))
            key = (score, -cur["rsi14"])
            if best is None or key > best[0]:
                best = (key, sym, ac, df)
        if best is None:
            return None
        _, sym, ac, df = best
        return Setup(sym, ac, "desk_pick", float(df["close"].iloc[-1]), str(df.index[-1]),
                     snapshot(df), derive_context(df))


class Ultron:
    def __init__(self, router: LLMRouter, lessons: LessonStore, scores: Scoreboard,
                 shadows: ShadowTracker):
        self.router, self.lessons, self.scores, self.shadows = router, lessons, scores, shadows

    async def _vote(self, agent: str, system: str, payload: dict,
                    prefer: tuple[str, ...], fmt: str = VOTE_FORMAT) -> Vote | None:
        user = json.dumps(payload, default=str, indent=1) + "\n\n" + fmt
        res = await self.router.ask(system + "\n\n" + DESK_RULES, user, prefer=prefer)
        if res is None:
            return None
        raw, provider = res
        return parse_vote(agent, raw, provider)

    def _hold(self, setup: Setup, votes, score, reason, lesson_ids) -> Verdict:
        return Verdict(Signal(setup.symbol, "HOLD", 0.0, reason), votes, score, reason, lesson_ids)

    async def _specialists(self, lead: str, base_prompt: str, brief: dict,
                           prefer: tuple[str, ...]) -> list[Vote]:
        """Both specialists under a lead answer the same brief, independently."""
        got = await asyncio.gather(*(
            self._vote(name, f"{base_prompt} {lens}", brief, prefer) for name, lens in SUBS[lead]))
        return [v for v in got if v]

    async def _lead(self, lead: str, base_prompt: str, subs: list[Vote], brief: dict,
                    prefer: tuple[str, ...]) -> Vote | None:
        """The lead reads both specialist answers and sides with the better one."""
        if not subs:
            return None
        payload = {**brief, "specialist_answers": [
            {**asdict(v), "track_record": self.scores.reliability(v.agent)} for v in subs]}
        vote = await self._vote(lead, LEAD_PROMPT.format(lead=lead) + " " + base_prompt,
                                payload, prefer, LEAD_FORMAT)
        if vote is not None:
            if vote.winner not in SUB_NAMES[lead]:
                vote.winner = ""
            return vote
        # The lead's own call failed. Fall back to the cautious reading of its specialists:
        # BUY only if every one of them said BUY, at the lowest confidence among them.
        buy = all(v.action == "BUY" for v in subs) and len(subs) == len(SUBS[lead])
        pick = min(subs, key=lambda v: v.confidence) if buy else max(subs, key=lambda v: v.confidence)
        return Vote(lead, "BUY" if buy else "HOLD", pick.confidence if buy else 0.0, pick.thesis,
                    pick.key_risk, "fallback", winner=pick.agent)

    async def deliberate(self, setup: Setup) -> Verdict:
        lessons = self.lessons.relevant(setup.symbol, setup.asset_class, setup.setup_type,
                                        setup.regime, k=3)
        lesson_ids = [l["id"] for l in lessons]
        brief = {
            "symbol": setup.symbol, "asset_class": setup.asset_class,
            "setup": setup.setup_type, "market_regime": setup.regime,
            "price_eur": setup.price_eur, "indicators_on_last_closed_1h_bar": setup.indicators,
            "derived_metrics": setup.context,
            "desk_history_with_this_setup": self.shadows.setup_stats(setup.setup_type, setup.regime),
            "lessons_from_past_trades": [l["lesson"] for l in lessons] or "none yet",
        }
        analyst_prompt = ANALYST_PROMPTS[setup.asset_class]
        # Spread the work over every free provider that is switched on; missing ones are skipped.
        gem_first = ("gemini", "cerebras", "grok")
        grok_first = ("grok", "cerebras", "gemini")
        cer_first = ("cerebras", "gemini", "grok")      # the bear argues on a different model

        # Stage 1: analyst and bull specialists answer in parallel.
        a_subs, b_subs = await asyncio.gather(
            self._specialists("analyst", analyst_prompt, brief, gem_first),
            self._specialists("bull", BULL_PROMPT, brief, grok_first))
        # Stage 2: those two leads pick their winners.
        analyst, bull = await asyncio.gather(
            self._lead("analyst", analyst_prompt, a_subs, brief, gem_first),
            self._lead("bull", BULL_PROMPT, b_subs, brief, grok_first))
        # Stage 3: the bear's specialists attack the bull lead's argument, then the bear lead picks.
        bear_brief = {**brief, "bull_argument": bull.thesis if bull else "(bull unavailable)"}
        r_subs = await self._specialists("bear", BEAR_PROMPT, bear_brief, cer_first)
        bear = await self._lead("bear", BEAR_PROMPT, r_subs, bear_brief, cer_first)

        leads = [v for v in (analyst, bull, bear) if v]
        votes = [*a_subs, *b_subs, *r_subs, *leads]
        if not votes:
            return self._hold(setup, votes, 0.0, "no agent answered: the AI provider failed or is out "
                              "of free quota, so this counts as HOLD", lesson_ids)

        # Weighted council of the three leads. A missing vote counts as HOLD.
        weights = {name: self.scores.reliability(name) for name in COUNCIL}
        buy = [v for v in leads if v.action == "BUY"]
        score = sum(weights[v.agent] * v.confidence for v in buy) / sum(weights.values())
        if len(buy) < 2 or score < COUNCIL_MIN_SCORE:
            return self._hold(setup, votes, score,
                              f"council: {len(buy)}/3 BUY, weighted score {score:.2f}", lesson_ids)

        # Ultron only gets asked when the council already leans BUY. Saves free-tier calls.
        ultron_brief = {
            **brief,
            "council_votes": [{**asdict(v), "track_record": weights[v.agent]} for v in leads],
            "specialist_votes": [{**asdict(v), "track_record": self.scores.reliability(v.agent)}
                                 for v in votes if v not in leads],
            "agent_scoreboard": self.scores.all() or "no graded calls yet",
        }
        final = await self._vote("ultron", ULTRON_PROMPT, ultron_brief, gem_first)
        if final is None:
            return self._hold(setup, votes, score, "Ultron unavailable, defaulting to HOLD", lesson_ids)
        votes.append(final)
        signal = Signal(setup.symbol, final.action, final.confidence, final.thesis)
        return Verdict(signal, votes, score, f"Ultron: {final.action}", lesson_ids)


class Historian:
    def __init__(self, router: LLMRouter, lessons: LessonStore):
        self.router, self.lessons = router, lessons

    async def post_mortem(self, trade: ClosedTrade, now: datetime) -> int:
        m = trade.meta or {}
        outcome = "win" if trade.pnl_eur > 0 else "loss"
        payload = {
            "symbol": trade.symbol, "setup": m.get("setup_type"), "regime": m.get("regime"),
            "practice_trade": bool(m.get("practice")),
            "entry_thesis": trade.thesis, "votes": m.get("votes"),
            "indicators_at_entry": m.get("indicators"),
            "entry_price": trade.entry_price, "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason, "return_pct": round(trade.return_pct, 3),
            "opened_at": trade.opened_at, "closed_at": trade.closed_at,
        }
        res = await self.router.ask(HISTORIAN_PROMPT, json.dumps(payload, default=str, indent=1),
                                    prefer=("gemini", "grok"))
        text = str(res[0].get("lesson", "")).strip() if res else ""
        if not text:  # no budget left: still keep a factual record
            text = (f"{m.get('setup_type')} on {trade.symbol} in {m.get('regime')} ended in a "
                    f"{outcome} ({trade.return_pct:+.2f}%, {trade.exit_reason}).")
        lesson_id = self.lessons.add(trade.symbol, trade.asset_class, m.get("setup_type"),
                                     m.get("regime"), outcome, trade.return_pct, text, now)
        log.info("Historian lesson #%d: %s", lesson_id, text)
        return lesson_id
