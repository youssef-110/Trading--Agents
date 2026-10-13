"""Central settings.

Every risk number lives here and nowhere else. No agent, Ultron included,
can change any of these at runtime.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional: read a .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct: float = 0.20        # share of current equity per trade (0.20 = 20%)
    stop_loss_pct: float = 2.0            # hard stop, not overridable
    take_profit_pct: float = 4.0          # 2:1 reward/risk
    daily_loss_limit_pct: float = 5.0     # drop from the 24h equity high that trips the breaker
    circuit_breaker_pause_hours: float = 24.0
    max_open_positions: int = 3
    min_confidence: float = 0.60          # Ultron's confidence must clear this
    min_order_eur: float = 1.0
    reentry_cooldown_hours: float = 6.0   # no new entry on a symbol for this long after a stop-loss


@dataclass(frozen=True)
class PracticeMode:
    """Small, labelled trades so the desk has real closed trades to learn from.

    Only when the full debate ends in HOLD but a lead still voted BUY. Same -2% / +4% exits,
    just a much smaller size and a lower bar than a real entry. These numbers live here, so
    no model can change them. PRACTICE_MODE=0 turns the whole thing off.
    """
    enabled: bool = field(default_factory=lambda: os.getenv("PRACTICE_MODE", "1") != "0")
    position_pct: float = 0.05            # share of equity per practice trade (5% of EUR 500 = EUR 25)
    min_lead_confidence: float = 0.50     # a lead's BUY has to be at least this confident
    max_open: int = 2                     # practice trades open at once (inside the overall max of 3)


@dataclass(frozen=True)
class ExecutionCosts:
    fee_rate: float = 0.001               # 0.1% per side
    slippage_rate: float = 0.0005         # 0.05% against us on every fill


@dataclass(frozen=True)
class ScoutRules:
    """Cheap deterministic filter. The LLM agents only wake up when one of these fires."""
    rsi_oversold: float = 35.0
    breakout_volume_ratio: float = 1.3
    max_atr_pct: float = 1.5              # if hourly ATR is bigger than this, a 2% stop is just noise
    # Quiet hours: every N hours with no setup, the desk debates the best-looking coin anyway
    # (setup type "desk_pick"). DESK_PICK_HOURS=0 turns it off. It never eats the last calls of the day.
    desk_pick_hours: int = field(default_factory=lambda: _env_int("DESK_PICK_HOURS", 2))
    desk_pick_reserve_calls: int = 40


@dataclass(frozen=True)
class LLMSettings:
    # Gemini: free tier, rate-limited. Keep billing OFF on the Google project.
    gemini_api_key: str | None = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    # Tried in order when the main model is rate-limited, slow, or gone. Comma-separated.
    gemini_fallback_models: str = field(default_factory=lambda: os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-flash-lite-latest,gemini-flash-latest"))
    gemini_daily_calls: int = field(default_factory=lambda: _env_int("GEMINI_DAILY_CALLS", 200))
    gemini_per_minute: int = field(default_factory=lambda: _env_int("GEMINI_PER_MINUTE", 5))

    # Grok: only promo credits are free. Never add a card in the xAI console.
    # Set XAI_DAILY_CALLS=0 to switch Grok off completely.
    xai_api_key: str | None = field(default_factory=lambda: os.getenv("XAI_API_KEY"))
    xai_model: str = field(default_factory=lambda: os.getenv("XAI_MODEL", "grok-4.3"))
    xai_daily_calls: int = field(default_factory=lambda: _env_int("XAI_DAILY_CALLS", 40))
    xai_per_minute: int = field(default_factory=lambda: _env_int("XAI_PER_MINUTE", 10))


    # Cerebras: free tier. Never add a card there either. Set CEREBRAS_DAILY_CALLS=0 to switch it off.
    cerebras_api_key: str | None = field(default_factory=lambda: os.getenv("CEREBRAS_API_KEY"))
    cerebras_model: str = field(default_factory=lambda: os.getenv("CEREBRAS_MODEL", "gpt-oss-120b"))
    cerebras_daily_calls: int = field(default_factory=lambda: _env_int("CEREBRAS_DAILY_CALLS", 300))
    cerebras_per_minute: int = field(default_factory=lambda: _env_int("CEREBRAS_PER_MINUTE", 20))


@dataclass(frozen=True)
class Settings:
    starting_balance_eur: float = 500.0
    db_path: str = field(default_factory=lambda: os.getenv("TRADER_DB", "paper_ledger.db"))
    risk: RiskLimits = field(default_factory=RiskLimits)
    practice: PracticeMode = field(default_factory=PracticeMode)
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)
    scout: ScoutRules = field(default_factory=ScoutRules)
    llm: LLMSettings = field(default_factory=LLMSettings)

    # Kraken lists EUR pairs directly, so crypto needs no FX conversion.
    crypto_exchange: str = field(default_factory=lambda: os.getenv("CRYPTO_EXCHANGE", "kraken"))
    crypto_symbols: tuple[str, ...] = (
        "BTC/EUR", "ETH/EUR", "SOL/EUR", "XRP/EUR", "ADA/EUR", "DOGE/EUR",
        "LINK/EUR", "LTC/EUR", "AVAX/EUR", "DOT/EUR", "ATOM/EUR", "UNI/EUR")
    crypto_timeframe: str = "1h"

    # US stocks are quoted in USD and converted to EUR in data_feed.py.
    stock_symbols: tuple[str, ...] = (
        "AAPL", "NVDA", "SPY", "QQQ", "MSFT", "TSLA", "AMD", "META", "AMZN", "GOOGL")
    stock_interval: str = "1h"

    cycle_seconds: int = 300              # how often the loop wakes up
    # Render kills the service when /health stays red for a while, and a restart used to wipe the
    # ledger. So a cycle has to stay short: a few debates at most (the rest wait for the next
    # cycle), and a debate that drags on because Gemini is busy gets dropped as a HOLD.
    max_debates_per_cycle: int = field(default_factory=lambda: _env_int("MAX_DEBATES_PER_CYCLE", 2))
    debate_timeout_s: float = field(default_factory=lambda: float(_env_int("DEBATE_TIMEOUT_S", 480)))
    live_price_seconds: int = 15          # how often crypto prices refresh for the dashboard only
    shadow_horizon_hours: int = 48        # how long a shadow setup gets to hit TP or SL
    log_path: str = "trader.log"
    report_path: str = "report.json"


SETTINGS = Settings()
