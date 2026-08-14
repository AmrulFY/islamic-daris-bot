from __future__ import annotations

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
