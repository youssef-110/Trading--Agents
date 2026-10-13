"""LLM access with a hard free-tier budget.

How this keeps spend at zero:
  * Every call is counted in SQLite, per provider per day, BEFORE it's sent.
    Once the daily cap is hit that provider simply isn't asked any more, and
    the desk defaults to HOLD.
  * Gemini's free quota resets at midnight Pacific time, so that's the day
    boundary used here for both providers.
  * If a provider answers "unauthorised / out of credit", it's switched off
    until the next day and the other provider takes over.

The real guarantee is on your side: never add a payment method to the Google
AI Studio project or the xAI console. Then the worst case is an error, never a bill.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from config import LLMSettings

log = logging.getLogger("llm")
PACIFIC = ZoneInfo("America/Los_Angeles")


class LLMError(Exception):
    pass


class RateLimited(LLMError):
    pass


class Overloaded(LLMError):
    """HTTP 5xx: the provider is busy right now. Worth one retry a few seconds later."""


class ProviderDisabled(LLMError):
    """Bad key, no credits left, or access denied. Don't retry today."""


def quota_day(now: datetime | None = None) -> str:
    return (now or datetime.now(PACIFIC)).astimezone(PACIFIC).date().isoformat()


def parse_json(text: str) -> dict:
    """Pull a JSON object out of a model reply, tolerating code fences and chatter."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"no JSON in reply: {text[:120]!r}")
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"broken JSON in reply: {text[:120]!r}") from e
    if not isinstance(obj, dict):
        raise LLMError("reply JSON is not an object")
    return obj


def _check(r: httpx.Response, provider: str) -> None:
    body = r.text[:300]
    if r.status_code == 429:
        raise RateLimited(f"{provider}: rate limited")
    if r.status_code in (401, 402, 403) or (r.status_code == 400 and "API_KEY" in body):
        raise ProviderDisabled(f"{provider}: HTTP {r.status_code} {body}")
    if r.status_code in (500, 502, 503, 504):
        raise Overloaded(f"{provider}: HTTP {r.status_code} {body[:120]}")
    if r.status_code >= 400:
        raise LLMError(f"{provider}: HTTP {r.status_code} {body}")


class GeminiProvider:
    """Google AI Studio. Tries the configured model first, then the fallbacks.

    Free-tier models come and go and some get very slow or run out of quota during the day,
    so a busy, missing or timed-out model just hands the call to the next one.
    """
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, model: str, http: httpx.AsyncClient,
                 fallbacks: tuple[str, ...] = (), timeout: float = 60):
        self.api_key, self.http, self.timeout = api_key, http, timeout
        self.models = [model, *[m for m in fallbacks if m and m != model]]
        self.model = model

    async def _one(self, model: str, system: str, user: str, temperature: float) -> dict:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        r = await self.http.post(self.URL.format(model=model),
                                 headers={"x-goog-api-key": self.api_key}, json=body, timeout=self.timeout)
        if r.status_code == 404:
            raise Overloaded(f"gemini: model {model} not available")
        _check(r, self.name)
        data = r.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError) as e:
            raise LLMError(f"gemini: unexpected response {str(data)[:200]}") from e
        return parse_json(text)

    async def complete_json(self, system: str, user: str, temperature: float = 0.4) -> dict:
        last: Exception | None = None
        for model in self.models:
            try:
                return await self._one(model, system, user, temperature)
            except (RateLimited, Overloaded, httpx.TimeoutException) as e:
                log.info("gemini %s unavailable (%s), trying the next model", model, type(e).__name__)
                last = e
        if isinstance(last, RateLimited):
            raise last
        raise Overloaded(f"gemini: no model answered ({type(last).__name__}: {last})")


class OpenAICompatProvider:
    """Any host that speaks the OpenAI chat-completions format."""
    name = ""
    URL = ""

    def __init__(self, api_key: str, model: str, http: httpx.AsyncClient):
        self.api_key, self.model, self.http = api_key, model, http

    async def complete_json(self, system: str, user: str, temperature: float = 0.4) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        r = await self.http.post(self.URL, headers={"Authorization": f"Bearer {self.api_key}"},
                                 json=body, timeout=90)
        _check(r, self.name)
        data = r.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"{self.name}: unexpected response {str(data)[:200]}") from e
        return parse_json(text)


class GrokProvider(OpenAICompatProvider):
    """xAI. Promo credits only."""
    name = "grok"
    URL = "https://api.x.ai/v1/chat/completions"


class CerebrasProvider(OpenAICompatProvider):
    """Cerebras Inference, free tier. Keys start with csk-."""
    name = "cerebras"
    URL = "https://api.cerebras.ai/v1/chat/completions"


class CallBudget:
    """Persistent per-provider daily call counter plus an in-memory per-minute window."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.db.execute("CREATE TABLE IF NOT EXISTS llm_calls ("
                        "provider TEXT NOT NULL, day TEXT NOT NULL, calls INTEGER NOT NULL, "
                        "PRIMARY KEY (provider, day))")
        self.db.commit()
        self._recent: dict[str, deque] = {}

    def used(self, provider: str, day: str | None = None) -> int:
        row = self.db.execute("SELECT calls FROM llm_calls WHERE provider = ? AND day = ?",
                              (provider, day or quota_day())).fetchone()
        return row[0] if row else 0

    def seconds_until_slot(self, provider: str, per_minute: int) -> float:
        q = self._recent.setdefault(provider, deque())
        now = time.monotonic()
        while q and now - q[0] >= 60:
            q.popleft()
        return 0.0 if len(q) < per_minute else 60 - (now - q[0]) + 0.05

    def take(self, provider: str, daily_limit: int, day: str | None = None) -> bool:
        day = day or quota_day()
        if self.used(provider, day) >= daily_limit:
            return False
        with self.db:
            self.db.execute("INSERT INTO llm_calls (provider, day, calls) VALUES (?, ?, 1) "
                            "ON CONFLICT(provider, day) DO UPDATE SET calls = calls + 1",
                            (provider, day))
        self._recent.setdefault(provider, deque()).append(time.monotonic())
        return True


class LLMRouter:
    """Sends a prompt to the first provider that's available and under budget."""

    def __init__(self, providers: dict, budget: CallBudget, limits: dict[str, tuple[int, int]]):
        self.providers = providers
        self.budget = budget
        self.limits = limits                     # name -> (daily_calls, per_minute)
        self._disabled_on: dict[str, str] = {}
        self.retry_wait_s = 4.0
        self.rate_wait_s = 25.0                  # free tiers throttle bursts; one patient retry usually works

    def available(self, name: str) -> bool:
        return name in self.providers and self._disabled_on.get(name) != quota_day()

    def can_afford(self, calls: int) -> bool:
        """True if some available provider still has at least this many calls left today."""
        return sum(max(0, self.limits[n][0] - self.budget.used(n))
                   for n in self.providers if self.available(n)) >= calls

    async def ask(self, system: str, user: str, prefer: tuple[str, ...] = ("gemini", "grok")):
        """Returns (parsed_json, provider_name), or None if nobody could answer."""
        order = list(prefer) + [n for n in self.providers if n not in prefer]
        for name in order:
            if not self.available(name):
                continue
            daily, per_minute = self.limits[name]
            wait = self.budget.seconds_until_slot(name, per_minute)
            if wait > 0:
                await asyncio.sleep(wait)
            if not self.budget.take(name, daily):
                log.info("%s daily budget (%d calls) used up", name, daily)
                continue
            try:
                try:
                    return await self.providers[name].complete_json(system, user), name
                except (Overloaded, RateLimited) as e:   # busy, not broken: one retry, which costs a call
                    log.info("%s busy (%s), retrying once", name, e)
                    await asyncio.sleep(self.rate_wait_s if isinstance(e, RateLimited) else self.retry_wait_s)
                    if not self.budget.take(name, daily):
                        continue
                    return await self.providers[name].complete_json(system, user), name
            except ProviderDisabled as e:
                log.warning("%s switched off until tomorrow: %s", name, e)
                self._disabled_on[name] = quota_day()
            except (LLMError, httpx.HTTPError, ValueError) as e:
                log.warning("%s call failed: %s", name, e)
        return None

    def usage(self) -> dict:
        return {name: {"used_today": self.budget.used(name),
                       "daily_limit": self.limits[name][0],
                       "available": self.available(name)}
                for name in self.providers}


def build_providers(cfg: LLMSettings, http: httpx.AsyncClient):
    providers, limits = {}, {}
    if cfg.gemini_api_key and cfg.gemini_daily_calls > 0:
        providers["gemini"] = GeminiProvider(cfg.gemini_api_key, cfg.gemini_model, http,
                                             tuple(m.strip() for m in cfg.gemini_fallback_models.split(",")))
        limits["gemini"] = (cfg.gemini_daily_calls, cfg.gemini_per_minute)
    if cfg.xai_api_key and cfg.xai_daily_calls > 0:
        providers["grok"] = GrokProvider(cfg.xai_api_key, cfg.xai_model, http)
        limits["grok"] = (cfg.xai_daily_calls, cfg.xai_per_minute)
    if cfg.cerebras_api_key and cfg.cerebras_daily_calls > 0:
        providers["cerebras"] = CerebrasProvider(cfg.cerebras_api_key, cfg.cerebras_model, http)
        limits["cerebras"] = (cfg.cerebras_daily_calls, cfg.cerebras_per_minute)
    return providers, limits
