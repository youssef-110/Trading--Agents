"""Market data + technical indicators.

Crypto comes from ccxt (public endpoints, no keys). Stocks come from yfinance
and get converted to EUR. ccxt and yfinance are imported lazily so the
indicator code and tests run without them installed.

Heads-up: yfinance scrapes Yahoo, is unofficial, and intraday data can lag.
It's fine for paper trading and prototyping, not for anything time-critical.
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
import pandas as pd

from config import Settings

# ---------------------------------------------------------------------------
# Indicators (pure pandas, no network)
# ---------------------------------------------------------------------------


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
    out = out.where(avg_loss != 0, 100.0)                         # only gains
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)   # flat
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns open/high/low/close/volume. Returns a copy with indicators added."""
    out = df.copy()
    c = out["close"]
    out["ema20"], out["ema50"], out["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    out["rsi14"] = rsi(c, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / c * 100
    vol_avg = out["volume"].rolling(20).mean()
    out["volume_ratio"] = out["volume"] / vol_avg.replace(0, np.nan)
    return out


def snapshot(df: pd.DataFrame) -> dict:
    """Latest bar's indicators as a compact dict, ready to drop into a prompt."""
    last = df.iloc[-1]
    keys = ["close", "ema20", "ema50", "ema200", "rsi14", "macd", "macd_signal",
            "macd_hist", "atr_pct", "volume_ratio"]
    snap = {k: (None if pd.isna(last[k]) else round(float(last[k]), 6)) for k in keys}
    snap["bars"] = len(df)
    snap["as_of"] = str(df.index[-1])
    return snap


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


class MarketData:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._exchange = None
        self._fx_cache: tuple[float, float] | None = None   # (eur_per_usd, fetched_at)

    def _crypto_exchange(self):
        if self._exchange is None:
            import ccxt
            self._exchange = getattr(ccxt, self.settings.crypto_exchange)({"enableRateLimit": True})
        return self._exchange

    async def crypto_ohlcv(self, symbol: str, timeframe: str | None = None,
                           limit: int = 300) -> pd.DataFrame:
        tf = timeframe or self.settings.crypto_timeframe
        ex = self._crypto_exchange()
        rows = await asyncio.to_thread(ex.fetch_ohlcv, symbol, tf, None, limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        return df

    async def live_prices(self, symbols) -> dict[str, float]:
        """Latest crypto prices in one request. Only for the dashboard's live numbers."""
        ex = self._crypto_exchange()
        tickers = await asyncio.to_thread(ex.fetch_tickers, list(symbols))
        return {s: float(t["last"]) for s, t in tickers.items() if t.get("last")}

    async def stock_ohlcv(self, ticker: str, interval: str | None = None,
                          period: str = "60d") -> pd.DataFrame:
        import yfinance as yf
        iv = interval or self.settings.stock_interval
        raw = await asyncio.to_thread(
            lambda: yf.Ticker(ticker).history(interval=iv, period=period, auto_adjust=False))
        if raw.empty:
            raise RuntimeError(f"no data for {ticker}")
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index = df.index.tz_convert("UTC")
        return df

    async def eur_per_usd(self, max_age_s: float = 600) -> float:
        if self._fx_cache and time.time() - self._fx_cache[1] < max_age_s:
            return self._fx_cache[0]
        import yfinance as yf
        hist = await asyncio.to_thread(
            lambda: yf.Ticker("EURUSD=X").history(period="5d", interval="1h"))
        usd_per_eur = float(hist["Close"].dropna().iloc[-1])
        rate = 1 / usd_per_eur
        self._fx_cache = (rate, time.time())
        return rate

    async def bars_eur(self, symbol: str, asset_class: str) -> pd.DataFrame:
        """OHLCV in EUR with indicators attached.

        USD stocks are converted at the *current* EUR/USD rate. That's a
        simplification for older bars, but RSI/MACD/EMA crossovers barely
        care, and the ledger only ever uses the latest price.
        """
        if asset_class == "crypto":
            df = await self.crypto_ohlcv(symbol)
            if not symbol.endswith("/EUR"):
                raise ValueError(f"{symbol}: use EUR-quoted pairs so the ledger stays in EUR")
        elif asset_class == "stock":
            df = await self.stock_ohlcv(symbol)
            fx = await self.eur_per_usd()
            df[["open", "high", "low", "close"]] *= fx
        else:
            raise ValueError(f"unsupported asset class: {asset_class}")
        return add_indicators(df)
