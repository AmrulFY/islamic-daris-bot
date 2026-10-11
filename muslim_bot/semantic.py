"""Semantic (embedding) fallback for passages that keyword search can't find.

Vectors live in their own SQLite file (never in the Go-managed databases) and
are built offline by `python -m muslim_bot.semantic_index`. At query time one
embeddings call turns the question into a vector and numpy ranks it against
the in-memory matrix. Any failure degrades to "no semantic results".
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from .retrieval import MAX_CONTEXT_PER_SOURCE, MAX_HADITH_CHARS, Passage, hadith_matn

logger = logging.getLogger(__name__)

EMBED_TIMEOUT_SECONDS = 4.0
MIN_SIMILARITY = 0.2
# While the index is missing or still being built, look again at most this often.
RELOAD_INTERVAL_SECONDS = 30.0

def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "create table if not exists embeddings ("
            " kind text not null, ref_id integer not null, vec blob not null,"
            " primary key (kind, ref_id))"
        )
        conn.execute("create table if not exists meta (key text primary key, value text not null)")
        conn.commit()


def mark_complete(db_path: Path, kind: str) -> None:
    """Record that every vector of `kind` has been written."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            "insert or replace into meta (key, value) values (?, datetime('now'))",
            (f"complete:{kind}",),
        )
        conn.commit()


def is_complete(db_path: Path, kind: str) -> bool:
    if not db_path.exists():
        return False
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute("select 1 from meta where key = ?", (f"complete:{kind}",)).fetchone()
    except sqlite3.DatabaseError:
        return False
    return row is not None


class SemanticSearcher:
    def __init__(
        self,
        *,
        db_path: Path,
        quran_db_path: Path,
        hadith_db_path: Path,
        model: str,
        dimensions: int,
    ) -> None:
        self.db_path = db_path
        self.quran_db_path = quran_db_path
        self.hadith_db_path = hadith_db_path
        self.model = model
        self.dimensions = dimensions
        self._lock = threading.Lock()
        self._last_load_attempt = 0.0
        self.building = False
        self._matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._client: Any = None
        self._client_key: tuple | None = None

    # -- index loading ---------------------------------------------------

    def _load(self, *, force: bool = False) -> None:
        """Load the vectors of every complete kind; retried while some are missing."""
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_load_attempt < RELOAD_INTERVAL_SECONDS:
                return
            self._last_load_attempt = now
            if not self.db_path.exists():  # connecting would create an empty file
                return
            try:
                with closing(sqlite3.connect(self.db_path)) as conn:
                    for kind in ("quran", "hadith"):
                        if kind in self._matrices or not is_complete(self.db_path, kind):
                            continue
                        rows = conn.execute(
                            "select ref_id, vec from embeddings where kind = ?", (kind,)
                        ).fetchall()
                        if not rows:
                            continue
                        ids = np.array([row[0] for row in rows], dtype=np.int64)
                        matrix = np.vstack(
                            [np.frombuffer(row[1], dtype=np.float32) for row in rows]
                        )
                        if matrix.shape[1] != self.dimensions:
                            logger.warning(
                                "semantic index %s has %d dims, expected %d; ignoring",
                                kind, matrix.shape[1], self.dimensions,
                            )
                            continue
                        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                        self._matrices[kind] = (ids, matrix / np.maximum(norms, 1e-9))
                        logger.info("semantic index loaded: %s=%d vectors", kind, len(ids))
            except Exception as exc:  # noqa: BLE001 - degrade to keyword-only.
                logger.warning("failed to load semantic index: %s", exc)

    def reload(self) -> None:
        """Pick up an index that was just (re)built."""
        with self._lock:
            self._matrices.clear()
        self._load(force=True)

    def status(self) -> str:
        """ready | building | missing."""
        self._load()
        if all(kind in self._matrices for kind in ("quran", "hadith")):
            return "ready"
        return "building" if self.building else "missing"

    def available(self, kind: str) -> bool:
        if kind not in self._matrices:
            self._load()
        return kind in self._matrices

    async def warm_up(self, *, api_key: str | None, base_url: str | None) -> None:
        """Load the index and open the gateway connection before the first request.

        The first embeddings call on a cold connection has taken ~16s, far past
        EMBED_TIMEOUT_SECONDS, so without this the first semantic request after
        a restart would silently get no semantic results.
        """
        if not await asyncio.to_thread(self.available, "quran"):
            return
        try:
            await self._embed("pemanasan", api_key=api_key, base_url=base_url)
        except Exception as exc:  # noqa: BLE001 - warm-up is best-effort.
            logger.warning("semantic warm-up failed: %s", exc)

    # -- querying --------------------------------------------------------

    async def _embed(self, text: str, *, api_key: str | None, base_url: str | None) -> np.ndarray:
        from openai import AsyncOpenAI

        key = (api_key, base_url)
        if self._client is None or self._client_key != key:
            self._client = AsyncOpenAI(
                api_key=api_key or "not-needed", base_url=base_url, timeout=EMBED_TIMEOUT_SECONDS
            )
            self._client_key = key
        response = await self._client.embeddings.create(
            model=self.model, input=text, dimensions=self.dimensions
        )
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-9)

    def _rank(self, kind: str, query: np.ndarray) -> list[tuple[int, float]]:
        ids, matrix = self._matrices[kind]
        scores = matrix @ query
        top = np.argsort(-scores)[:MAX_CONTEXT_PER_SOURCE]
        return [(int(ids[i]), float(scores[i])) for i in top if scores[i] >= MIN_SIMILARITY]

    async def search(
        self,
        query: str,
        kinds: list[str],
        *,
        language: str,
        api_key: str | None,
        base_url: str | None,
    ) -> list[Passage]:
        kinds = [kind for kind in kinds if await asyncio.to_thread(self.available, kind)]
        if not kinds:
            return []
        try:
            vector = await asyncio.wait_for(
                self._embed(query, api_key=api_key, base_url=base_url),
                timeout=EMBED_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - semantic search is best-effort.
            logger.warning("embedding call failed, skipping semantic search: %s", exc)
            return []
        return await asyncio.to_thread(self._passages, kinds, vector, language)

    def _passages(self, kinds: list[str], vector: np.ndarray, language: str) -> list[Passage]:
        passages: list[Passage] = []
        try:
            if "quran" in kinds:
                passages += self._quran_passages(self._rank("quran", vector), language)
            if "hadith" in kinds:
                passages += self._hadith_passages(self._rank("hadith", vector), language)
        except Exception as exc:  # noqa: BLE001
            logger.warning("semantic passage lookup failed: %s", exc)
            return []
        return passages

    def _quran_passages(self, ranked: list[tuple[int, float]], language: str) -> list[Passage]:
        if not ranked:
            return []
        column = "a.translation_en" if language == "en" else "a.translation_indo"
        scores = dict(ranked)
        marks = ",".join("?" for _ in ranked)
        with closing(sqlite3.connect(self.quran_db_path)) as conn:
            rows = conn.execute(
                f"select a.id, a.surah_id, a.number_in_surah, {column}, s.name_transliteration"
                " from ayahs a join surahs s on s.id = a.surah_id"
                f" where a.id in ({marks})",
                [ref_id for ref_id, _ in ranked],
            ).fetchall()
        rows.sort(key=lambda row: -scores[row[0]])
        return [
            Passage(
                kind="quran",
                label=f"{name} {surah}:{ayah}",
                source_id=f"{surah}:{ayah}",
                text=str(text),
                score=scores[ref_id],
            )
            for ref_id, surah, ayah, text, name in rows
        ]

    def _hadith_passages(self, ranked: list[tuple[int, float]], language: str) -> list[Passage]:
        if not ranked:
            return []
        preferred = ["en", "id"] if language == "en" else ["id", "en"]
        scores = dict(ranked)
        marks = ",".join("?" for _ in ranked)
        with closing(sqlite3.connect(self.hadith_db_path)) as conn:
            rows = conn.execute(
                "select h.id, h.number, b.name_en, ht.lang, ht.text from hadiths h"
                " join books b on b.id = h.book_id"
                " join hadith_texts ht on ht.hadith_id = h.id and ht.lang in ('id', 'en')"
                f" where h.id in ({marks})",
                [ref_id for ref_id, _ in ranked],
            ).fetchall()
        grouped: dict[int, tuple[int, str, dict[str, str]]] = {}
        for ref_id, number, book, lang, text in rows:
            grouped.setdefault(ref_id, (number, book, {}))[2][lang] = str(text)
        passages: list[Passage] = []
        for ref_id, (number, book, texts) in sorted(grouped.items(), key=lambda item: -scores[item[0]]):
            text = next((texts[lang] for lang in preferred if texts.get(lang, "").strip()), "")
            if not text:
                continue
            passages.append(
                Passage(
                    kind="hadith",
                    label=f"{book} #{number}",
                    source_id=str(number),
                    text=hadith_matn(text, MAX_HADITH_CHARS),
                    score=scores[ref_id],
                )
            )
        return passages
