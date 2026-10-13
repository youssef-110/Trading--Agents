import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ExecutionCosts, RiskLimits  # noqa: E402
from data_feed import add_indicators, rsi, snapshot  # noqa: E402
from portfolio import InsufficientFunds, Portfolio  # noqa: E402
from risk_manager import Bar, RiskManager, Signal  # noqa: E402

T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
COSTS = ExecutionCosts()


@pytest.fixture
def pf(tmp_path):
    return Portfolio(str(tmp_path / "t.db"), 50.0, COSTS)


@pytest.fixture
def rm(pf):
    return RiskManager(pf, RiskLimits())


def buy(ticker="BTC/EUR", conf=0.8):
    return Signal(ticker, "BUY", conf, "test")


# ---------------- ledger ----------------

def test_open_close_math(pf):
    p = pf.open_position("BTC/EUR", "crypto", 100.0, 10.0, 2.0, 4.0, now=T0)
    assert p.entry_price == pytest.approx(100.05)
    assert pf.cash == pytest.approx(50 - 10 - 0.01)
    assert p.stop_price == pytest.approx(100.05 * 0.98)

    t = pf.close_position(p.id, 104.0, "manual", now=T0)
    proceeds = p.qty * 104.0 * (1 - COSTS.slippage_rate)
    expected_pnl = proceeds * (1 - COSTS.fee_rate) - 10.01
    assert t.pnl_eur == pytest.approx(expected_pnl)
    assert pf.cash == pytest.approx(50 + expected_pnl)
    assert pf.open_positions() == []
    assert pf.metrics()["trades"] == 1


def test_cannot_overspend(pf):
    with pytest.raises(InsufficientFunds):
        pf.open_position("BTC/EUR", "crypto", 100.0, 50.0, 2, 4)


def test_ledger_survives_restart(tmp_path):
    db = str(tmp_path / "r.db")
    a = Portfolio(db, 50.0, COSTS)
    a.open_position("ETH/EUR", "crypto", 2000.0, 10.0, 2, 4)
    b = Portfolio(db, 999.0, COSTS)  # starting balance ignored once initialised
    assert b.cash == pytest.approx(a.cash)
    assert len(b.open_positions()) == 1


# ---------------- risk ----------------

def test_sizing_is_20pct_of_equity(rm):
    d = rm.evaluate(buy(), 100.0, {}, now=T0)
    assert d.approved and d.notional_eur == 10.0
    assert (d.stop_loss_pct, d.take_profit_pct) == (2.0, 4.0)


def test_model_cannot_set_stop(rm):
    s = Signal("BTC/EUR", "BUY", 0.9, "x", suggested_stop_loss_pct=25.0)
    assert rm.evaluate(s, 100.0, {}, now=T0).stop_loss_pct == 2.0


def test_rejections(rm):
    assert not rm.evaluate(buy(conf=0.3), 100, {}).approved
    assert not rm.evaluate(Signal("BTC/EUR", "SELL", 0.9), 100, {}).approved
    assert not rm.evaluate(Signal("BTC/EUR", "HOLD", 0.9), 100, {}).approved
    for sym in ("A", "B", "C"):
        rm.execute(rm.evaluate(buy(sym), 100, {}), buy(sym), "crypto", 100)
    d = rm.evaluate(buy("D"), 100, {})
    assert not d.approved and "max open" in d.reason
    assert not rm.evaluate(buy("A"), 100, {}).approved  # already holding


def test_stop_and_take_profit(rm):
    rm.execute(rm.evaluate(buy("A"), 100, {}), buy("A"), "crypto", 100)
    rm.execute(rm.evaluate(buy("B"), 100, {}), buy("B"), "crypto", 100)
    closed = rm.check_exits({"A": Bar(last=97.0), "B": Bar(last=105.0)})
    assert {c.symbol: c.exit_reason for c in closed} == {"A": "stop_loss", "B": "take_profit"}
    loss = [c for c in closed if c.symbol == "A"][0]
    assert loss.return_pct < -2.0  # gap through the stop fills worse, plus fees


def test_no_reentry_right_after_a_stop(rm):
    rm.execute(rm.evaluate(buy("A"), 100, {}, now=T0), buy("A"), "crypto", 100, now=T0)
    rm.check_exits({"A": Bar(last=97.0)}, now=T0)
    d = rm.evaluate(buy("A"), 97, {}, now=T0 + timedelta(hours=1))
    assert not d.approved and "cooling off" in d.reason
    assert rm.evaluate(buy("A"), 97, {}, now=T0 + timedelta(hours=7)).approved


def test_both_levels_in_one_bar_assumes_stop(rm):
    rm.execute(rm.evaluate(buy("A"), 100, {}), buy("A"), "crypto", 100)
    closed = rm.check_exits({"A": Bar(last=100.0, low=90.0, high=110.0)})
    assert closed[0].exit_reason == "stop_loss"


def test_circuit_breaker_trips_and_persists(tmp_path):
    db = str(tmp_path / "cb.db")
    pf = Portfolio(db, 50.0, COSTS)
    rm = RiskManager(pf, RiskLimits())
    rm.execute(rm.evaluate(buy("A"), 100, {}), buy("A"), "crypto", 100)
    assert rm.update_circuit_breaker({"A": 100.0}, now=T0) is False
    assert rm.update_circuit_breaker({"A": 70.0}, now=T0 + timedelta(hours=1)) is True

    rm2 = RiskManager(Portfolio(db, 50.0, COSTS), RiskLimits())  # restart
    later = T0 + timedelta(hours=2)
    assert rm2.is_halted(later)
    assert not rm2.evaluate(buy("B"), 100, {}, now=later).approved
    assert rm2.evaluate(Signal("A", "SELL", 0.9), 70, {}, now=later).approved  # exits still allowed
    assert not rm2.is_halted(T0 + timedelta(hours=26))


def test_signal_parsing():
    s = Signal.from_dict({"ticker": "BTC/EUR", "action": "buy", "confidence": 0.7, "thesis": "x"})
    assert s.action == "BUY"
    for bad in ({"ticker": "X", "action": "YOLO", "confidence": 0.5},
                {"ticker": "X", "action": "BUY", "confidence": 1.5},
                {"ticker": "", "action": "BUY", "confidence": 0.5}):
        with pytest.raises(ValueError):
            Signal.from_dict(bad)


# ---------------- indicators ----------------

def ohlcv(close):
    close = pd.Series(close, dtype=float)
    return pd.DataFrame({"open": close, "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": 100.0})


def test_rsi_bounds_and_extremes():
    assert rsi(pd.Series(np.arange(1, 60, dtype=float))).iloc[-1] == pytest.approx(100.0)
    walk = pd.Series(100 + np.random.default_rng(0).normal(0, 1, 500).cumsum())
    r = rsi(walk).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_indicator_snapshot():
    df = add_indicators(ohlcv(100 + np.random.default_rng(1).normal(0, 1, 300).cumsum()))
    snap = snapshot(df)
    for k in ("rsi14", "ema200", "macd_hist", "atr_pct", "volume_ratio"):
        assert snap[k] is not None


# ---------------- practice mode ----------------

def test_practice_trades_are_small_and_capped(pf):
    from config import PracticeMode
    rm = RiskManager(pf, RiskLimits(), PracticeMode(enabled=True))
    d = rm.evaluate(buy(conf=0.6), 100.0, {}, now=T0, practice=True)
    assert d.approved and d.notional_eur == 2.5                    # 5% of EUR 50, not 20%
    assert (d.stop_loss_pct, d.take_profit_pct) == (2.0, 4.0)       # same exits as a real trade
    assert not rm.evaluate(buy(conf=0.45), 100.0, {}, now=T0, practice=True).approved  # lower bar, still a bar
    for sym in ("BTC/EUR", "ETH/EUR"):
        pf.open_position(sym, "crypto", 100.0, 2.5, 2, 4, meta={"practice": True}, now=T0)
    assert "practice" in rm.evaluate(buy("SOL/EUR", 0.9), 100.0, {}, now=T0, practice=True).reason


def test_practice_is_refused_when_switched_off(pf):
    from config import PracticeMode
    rm = RiskManager(pf, RiskLimits(), PracticeMode(enabled=False))
    assert not rm.evaluate(buy(conf=0.9), 100.0, {}, now=T0, practice=True).approved
    assert rm.evaluate(buy(conf=0.9), 100.0, {}, now=T0).approved   # normal entries are untouched


def test_untraded_ledger_takes_a_new_starting_balance(tmp_path):
    db = str(tmp_path / "b.db")
    Portfolio(db, 50.0, COSTS)
    assert Portfolio(db, 500.0, COSTS).cash == 500.0           # nothing traded yet: new balance applies
    traded = Portfolio(db, 500.0, COSTS)
    traded.open_position("BTC/EUR", "crypto", 100.0, 10.0, 2, 4)
    assert Portfolio(db, 5000.0, COSTS).starting_balance == 500.0   # history exists: left alone
