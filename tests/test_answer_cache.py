from __future__ import annotations

from pathlib import Path

from muslim_bot.answer_cache import AnswerCache, cache_key, normalize_message


def make_cache(tmp_path: Path, **kwargs) -> AnswerCache:
    return AnswerCache(tmp_path / "cache.db", **kwargs)


def sample_payload() -> dict:
    return {
        "references": [{"type": "hadith", "label": "Sahih al-Bukhari #1", "source_id": "1", "verified": True}],
        "used_tools": ["search_hadith"],
        "warnings": [],
    }


def test_roundtrip(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    cache.put("Hadis tentang niat", "id", "Jawaban.", sample_payload())

    hit = cache.get("hadis tentang niat", "id")

    assert hit is not None
    assert hit["answer"] == "Jawaban."
    assert hit["references"][0].label == "Sahih al-Bukhari #1"
    assert hit["references"][0].verified is True
    assert hit["used_tools"] == ["search_hadith"]


def test_key_normalization(tmp_path: Path) -> None:
    assert normalize_message("  Hadis   TENTANG Niat ") == "hadis tentang niat"
    assert cache_key("Hadis tentang niat", "id") == cache_key("  hadis TENTANG NIAT", "ID")


def test_language_isolates_entries(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    cache.put("halo", "id", "Jawaban ID.", sample_payload())

    assert cache.get("halo", "id") is not None
    assert cache.get("halo", "en") is None


def test_miss_returns_none(tmp_path: Path) -> None:
    assert make_cache(tmp_path).get("tidak ada", "id") is None


def test_expired_entries_miss(tmp_path: Path) -> None:
    cache = make_cache(tmp_path, ttl_days=-1)
    cache.put("halo", "id", "Jawaban.", sample_payload())

    assert cache.get("halo", "id") is None


def test_disabled_cache(tmp_path: Path) -> None:
    cache = make_cache(tmp_path, ttl_days=7)
    disabled = AnswerCache(tmp_path / "cache2.db", enabled=False)
    disabled.put("halo", "id", "Jawaban.", sample_payload())

    assert disabled.get("halo", "id") is None
    cache.put("halo", "id", "Jawaban.", sample_payload())
    assert cache.get("halo", "id") is not None


def test_purge_expired(tmp_path: Path) -> None:
    cache = make_cache(tmp_path, ttl_days=7)
    cache.put("baru", "id", "Jawaban.", sample_payload())

    assert cache.purge_expired() == 0

    old = AnswerCache(tmp_path / "old-cache.db", ttl_days=-1)
    old.put("lama", "id", "Jawaban.", sample_payload())

    assert old.purge_expired() == 1
    assert old.get("lama", "id") is None


def test_max_entries_bounded(tmp_path: Path) -> None:
    cache = make_cache(tmp_path, ttl_days=30, max_entries=3)
    for index in range(5):
        cache.put(f"pesan {index}", "id", f"Jawaban {index}.", sample_payload())

    import sqlite3

    count = sqlite3.connect(cache.db_path).execute("select count(*) from answer_cache").fetchone()[0]

    assert count == 3
