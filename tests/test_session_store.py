from __future__ import annotations

from pathlib import Path

from muslim_bot.session_store import SessionStore


def test_session_history_includes_recent_turns() -> None:
    store = SessionStore(max_turns=2)
    session_id = store.new_session_id()

    store.append(session_id, "satu", "jawab satu")
    store.append(session_id, "dua", "jawab dua")
    prompt = store.history_prompt(session_id, "tiga")

    assert "Pengguna: satu" in prompt
    assert "Asisten: jawab dua" in prompt
    assert "Pertanyaan terbaru: tiga" in prompt


def test_session_history_respects_max_turns() -> None:
    store = SessionStore(max_turns=1)
    session_id = store.new_session_id()

    store.append(session_id, "lama", "jawab lama")
    store.append(session_id, "baru", "jawab baru")
    prompt = store.history_prompt(session_id, "lanjut")

    assert "lama" not in prompt
    assert "baru" in prompt
    store.close()


def test_history_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    store_a = SessionStore(max_turns=3, db_path=db_path)
    session_id = store_a.new_session_id()
    store_a.append(session_id, "halo", "jawab halo")
    store_a.close()

    store_b = SessionStore(max_turns=3, db_path=db_path)
    prompt = store_b.history_prompt(session_id, "lanjut")

    assert "Pengguna: halo" in prompt
    assert store_b.ensure_session(session_id) == session_id
    store_b.close()


def test_purge_inactive_removes_stale_sessions_only(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(max_turns=3, db_path=db_path)
    old_id = store.ensure_session("session-lama")
    fresh_id = store.ensure_session("session-baru")

    with store._lock:
        store._conn.execute(
            "update sessions set updated_at = '2000-01-01T00:00:00+00:00' where id = ?",
            (old_id,),
        )
        store._conn.commit()

    removed = store.purge_inactive(days=7)

    assert removed == 1
    assert store.history_prompt(old_id, "uji") == "uji"
    assert store.history_prompt(fresh_id, "uji") == "uji"

    store.append(fresh_id, "masih hidup", "ya")
    assert "masih hidup" in store.history_prompt(fresh_id, "uji")
    store.close()


def test_store_creates_missing_parent_directories(tmp_path: Path) -> None:
    db_path = tmp_path / "nested" / "deep" / "sessions.db"
    assert not db_path.parent.exists()

    store = SessionStore(max_turns=3, db_path=db_path)
    session_id = store.new_session_id()
    store.append(session_id, "halo", "jawab halo")
    store.close()

    assert db_path.exists()
    reopened = SessionStore(max_turns=3, db_path=db_path)
    try:
        assert "halo" in reopened.history_prompt(session_id, "lanjut")
    finally:
        reopened.close()
