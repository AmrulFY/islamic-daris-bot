from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .schemas import Reference


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_message(message: str) -> str:
    return " ".join(message.strip().lower().split())


def cache_key(message: str, language: str) -> str:
    raw = f"{normalize_message(message)}|{language.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AnswerCache:
    """SQLite cache for agent answers.

    Only complete agent replies are cached (never fast-path lookups, which are
    already instant). Entries expire after `ttl_days`.
    """

    def __init__(
        self,
        db_path: str | Path,
        ttl_days: float = 7.0,
        enabled: bool = True,
        max_entries: int = 2000,
    ) -> None:
        self.db_path = Path(db_path)
        self.ttl_days = ttl_days
        self.enabled = enabled
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute(
                    """
                    create table if not exists answer_cache (
                        key text primary key,
                        message text not null,
                        language text not null,
                        answer text not null,
                        payload text not null,
                        created_at text not null,
                        updated_at text not null
                    )
                    """
                )
                conn.commit()

    def get(self, message: str, language: str) -> dict | None:
        if not self.enabled:
            return None

        key = cache_key(message, language)
        cutoff = (_utcnow() - timedelta(days=self.ttl_days)).isoformat(timespec="seconds")
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                row = conn.execute(
                    "select answer, payload, updated_at from answer_cache where key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return None
                if row[2] < cutoff:
                    conn.execute("delete from answer_cache where key = ?", (key,))
                    conn.commit()
                    return None
                payload = json.loads(row[1])
        return {
            "answer": row[0],
            "references": [
                Reference.model_validate(ref) for ref in payload.get("references", [])
            ],
            "used_tools": payload.get("used_tools", []),
            "warnings": payload.get("warnings", []),
        }

    def put(self, message: str, language: str, answer: str, payload: dict) -> None:
        if not self.enabled:
            return

        key = cache_key(message, language)
        now = _utcnow().isoformat(timespec="seconds")
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.execute(
                    "insert into answer_cache (key, message, language, answer, payload, created_at, updated_at)"
                    " values (?, ?, ?, ?, ?, ?, ?)"
                    " on conflict(key) do update set"
                    " answer = excluded.answer, payload = excluded.payload, updated_at = excluded.updated_at",
                    (key, message, language, answer, json.dumps(payload, ensure_ascii=False), now, now),
                )
                count = conn.execute("select count(*) from answer_cache").fetchone()[0]
                if count > self.max_entries:
                    conn.execute(
                        "delete from answer_cache where key not in ("
                        " select key from answer_cache order by updated_at desc limit ?"
                        ")",
                        (self.max_entries,),
                    )
                conn.commit()

    def purge_expired(self) -> int:
        cutoff = (_utcnow() - timedelta(days=self.ttl_days)).isoformat(timespec="seconds")
        with self._lock:
            with closing(sqlite3.connect(self.db_path)) as conn:
                cursor = conn.execute("delete from answer_cache where updated_at < ?", (cutoff,))
                removed = cursor.rowcount
                conn.commit()
        return removed
