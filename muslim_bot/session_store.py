from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


_SCHEMA = """
create table if not exists sessions (
    id text primary key,
    created_at text not null,
    updated_at text not null
);
create table if not exists turns (
    id integer primary key autoincrement,
    session_id text not null references sessions(id),
    user_text text not null,
    assistant_text text not null,
    created_at text not null
);
create index if not exists idx_turns_session on turns(session_id, id);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


class SessionStore:
    """Chat history store backed by SQLite (file or in-memory).

    With ``db_path=None`` an in-memory database is used, matching the old
    in-memory behavior. With a file path the history survives restarts and is
    shared across workers.
    """

    def __init__(self, max_turns: int = 12, db_path: str | Path | None = None) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        target = str(db_path) if db_path else ":memory:"
        if db_path:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(target, check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def new_session_id(self) -> str:
        session_id = uuid4().hex
        now = _iso(_utcnow())
        with self._lock:
            self._conn.execute(
                "insert into sessions (id, created_at, updated_at) values (?, ?, ?)",
                (session_id, now, now),
            )
            self._conn.commit()
        return session_id

    def ensure_session(self, session_id: str | None) -> str:
        if not session_id:
            return self.new_session_id()
        now = _iso(_utcnow())
        with self._lock:
            self._conn.execute(
                "insert or ignore into sessions (id, created_at, updated_at) values (?, ?, ?)",
                (session_id, now, now),
            )
            self._conn.commit()
        return session_id

    def append(self, session_id: str, user: str, assistant: str) -> None:
        now = _iso(_utcnow())
        with self._lock:
            self._ensure_session_row(session_id, now)
            self._conn.execute(
                "insert into turns (session_id, user_text, assistant_text, created_at)"
                " values (?, ?, ?, ?)",
                (session_id, user, assistant, now),
            )
            self._conn.execute(
                "delete from turns where session_id = ? and id not in ("
                "  select id from turns where session_id = ? order by id desc limit ?"
                ")",
                (session_id, session_id, self.max_turns),
            )
            self._conn.commit()

    def recent_turns(self, session_id: str, limit: int = 4) -> list[tuple[str, str]]:
        """Up to `limit` latest (user text, assistant text) pairs, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "select user_text, assistant_text from ("
                "  select id, user_text, assistant_text from turns"
                "  where session_id = ? order by id desc limit ?"
                ") order by id asc",
                (session_id, limit),
            ).fetchall()
        return [(str(user), str(assistant)) for user, assistant in rows]

    def last_turn(self, session_id: str) -> tuple[str, str] | None:
        """(user text, assistant text) of the most recent turn, if any."""
        with self._lock:
            row = self._conn.execute(
                "select user_text, assistant_text from turns where session_id = ? order by id desc limit 1",
                (session_id,),
            ).fetchone()
        return (str(row[0]), str(row[1])) if row else None

    def history_prompt(self, session_id: str, latest_message: str) -> str:
        with self._lock:
            rows = self._conn.execute(
                "select user_text, assistant_text from ("
                "  select id, user_text, assistant_text from turns"
                "  where session_id = ? order by id desc limit ?"
                ") order by id asc",
                (session_id, self.max_turns),
            ).fetchall()
        if not rows:
            return latest_message

        lines = [
            "Riwayat percakapan singkat untuk konteks. Jangan mengutip riwayat ini sebagai sumber agama.",
        ]
        for user_text, assistant_text in rows:
            lines.append(f"Pengguna: {user_text}")
            lines.append(f"Asisten: {assistant_text}")
        lines.append(f"Pertanyaan terbaru: {latest_message}")
        return "\n".join(lines)

    def purge_inactive(self, days: float = 7.0) -> int:
        cutoff = _iso(_utcnow() - timedelta(days=days))
        with self._lock:
            cursor = self._conn.execute("select count(*) from sessions where updated_at < ?", (cutoff,))
            stale_count = int(cursor.fetchone()[0])
            if stale_count == 0:
                return 0
            self._conn.execute("delete from turns where session_id in (select id from sessions where updated_at < ?)", (cutoff,))
            self._conn.execute("delete from sessions where updated_at < ?", (cutoff,))
            self._conn.commit()
        return stale_count

    def _ensure_session_row(self, session_id: str, now: str) -> None:
        self._conn.execute(
            "insert or ignore into sessions (id, created_at, updated_at) values (?, ?, ?)",
            (session_id, now, now),
        )
        self._conn.execute(
            "update sessions set updated_at = ? where id = ?",
            (now, session_id),
        )
