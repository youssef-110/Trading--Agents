"""Keeps the ledger alive on hosts with a throwaway disk (Render's free tier).

The database is the desk's memory: lessons, agent scores, shadow results, the 50 EUR
account. This copies it to a private GitHub repo (branch "state") and restores it on
startup when the local file is missing. Everything is optional: with no token the bot
just uses the local file, and a backup failure is only ever a warning.

Needs, in the environment:
  BACKUP_GITHUB_TOKEN   fine-grained token, "Contents: read and write" on that one repo
  BACKUP_REPO           owner/name of a PRIVATE repo
  BACKUP_BRANCH         optional, default "state"
"""
from __future__ import annotations

import base64
import gzip
import logging
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import httpx

log = logging.getLogger("backup")
API = "https://api.github.com"
FILE = "ledger.db.gz"
# What the backup is doing right now, shown on /health so a broken backup can't go unnoticed
# again (it silently never ran on Render until 2026-10-13). No secrets, no repo name.
STATUS = {"text": "not set up yet"}


def _say(text: str) -> None:
    STATUS["text"] = text


def _why(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    return type(e).__name__


class Backup:
    def __init__(self, http: httpx.AsyncClient | None, db_path: str, token: str | None = None,
                 repo: str | None = None, branch: str | None = None, min_gap_s: float = 3600):
        self.http, self.db_path = http, db_path
        self.token = token if token is not None else os.getenv("BACKUP_GITHUB_TOKEN", "")
        self.repo = repo if repo is not None else os.getenv("BACKUP_REPO", "")
        self.branch = branch or os.getenv("BACKUP_BRANCH", "state")
        self.min_gap_s, self._last = min_gap_s, 0.0
        self._dirty = True
        missing = [n for n, v in (("BACKUP_GITHUB_TOKEN", self.token), ("BACKUP_REPO", self.repo)) if not v]
        _say(f"off ({', '.join(missing)} not set)" if missing else "on, nothing saved yet")

    @property
    def enabled(self) -> bool:
        return bool(self.http and self.token and self.repo)

    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    async def _get_file(self) -> tuple[bytes, str] | None:
        r = await self.http.get(f"{API}/repos/{self.repo}/contents/{FILE}", params={"ref": self.branch},
                                headers=self._h(), timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        j = r.json()
        if j.get("content"):
            return base64.b64decode(j["content"]), j["sha"]
        raw = await self.http.get(j["download_url"], headers=self._h(), timeout=60)   # >1MB files
        raw.raise_for_status()
        return raw.content, j["sha"]

    async def restore_if_missing(self) -> bool:
        """Call before the database is opened."""
        if not self.enabled or (Path(self.db_path).exists() and Path(self.db_path).stat().st_size > 0):
            return False
        try:
            got = await self._get_file()
            if got is None:
                log.info("no backup in %s@%s yet, starting with a fresh ledger", self.repo, self.branch)
                _say("on, no earlier backup found, started fresh")
                return False
            Path(self.db_path).write_bytes(gzip.decompress(got[0]))
            log.info("restored the ledger from %s@%s", self.repo, self.branch)
            _say("on, ledger restored at startup")
            return True
        except Exception as e:
            log.warning("restore failed, starting fresh: %s", e)
            _say(f"FAILING, restore at startup failed ({_why(e)})")
            return False

    def mark_dirty(self) -> None:
        self._dirty = True

    def _snapshot(self) -> bytes:
        """A consistent copy of the live database, compressed."""
        with tempfile.TemporaryDirectory() as d:
            tmp = os.path.join(d, "snap.db")
            src, dst = sqlite3.connect(self.db_path), sqlite3.connect(tmp)
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            return gzip.compress(Path(tmp).read_bytes(), 6)

    async def _ensure_branch(self) -> None:
        r = await self.http.get(f"{API}/repos/{self.repo}/git/ref/heads/{self.branch}", headers=self._h(), timeout=30)
        if r.status_code != 404:
            r.raise_for_status()
            return
        info = await self.http.get(f"{API}/repos/{self.repo}", headers=self._h(), timeout=30)
        info.raise_for_status()
        base = await self.http.get(f"{API}/repos/{self.repo}/git/ref/heads/{info.json()['default_branch']}",
                                   headers=self._h(), timeout=30)
        base.raise_for_status()
        made = await self.http.post(f"{API}/repos/{self.repo}/git/refs", headers=self._h(), timeout=30,
                                    json={"ref": f"refs/heads/{self.branch}", "sha": base.json()["object"]["sha"]})
        made.raise_for_status()

    async def maybe_backup(self, force: bool = False) -> bool:
        """Cheap to call every cycle: it only uploads when something changed and the gap has passed."""
        if not self.enabled or not (self._dirty or force):
            return False
        if not force and time.monotonic() - self._last < self.min_gap_s:
            return False
        try:
            await self._ensure_branch()
            current = await self._get_file()
            body = {"message": "ledger backup", "branch": self.branch,
                    "content": base64.b64encode(self._snapshot()).decode()}
            if current:
                body["sha"] = current[1]
            r = await self.http.put(f"{API}/repos/{self.repo}/contents/{FILE}", headers=self._h(),
                                    json=body, timeout=60)
            r.raise_for_status()
            self._last, self._dirty = time.monotonic(), False
            _say(f"ok, last saved {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC")
            return True
        except Exception as e:
            log.warning("backup failed (will retry): %s", e)
            _say(f"FAILING ({_why(e)}), retrying")
            self._last = time.monotonic() - self.min_gap_s + 60     # retry in about a minute
            return False
