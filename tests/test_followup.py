from __future__ import annotations

from pathlib import Path

import pytest

from muslim_bot.retrieval import Passage, build_retrieval_query, extract_keywords, format_context
from muslim_bot.session_store import SessionStore
from tests.test_retrieval import make_dbs
from tests.test_streaming import collect, delta, install_agents_stub, make_runner

PREVIOUS = [
    (
        "ayat tentang riba",
        "Allah melarang riba. Rujukan: Al-Baqarah 2:278 dan Sahih al-Bukhari #1.",
    )
]


def test_follow_up_without_keywords_borrows_previous_topic() -> None:
    query = build_retrieval_query("jelaskan ayat tadi", "ayat tentang riba")

    assert query.follow_up
    assert "riba" in query.text
    assert query.sources_from == "jelaskan ayat tadi"


def test_follow_up_naming_a_source_decides_the_source() -> None:
    query = build_retrieval_query("bagaimana dengan hadits?", "ayat tentang riba")

    assert query.follow_up and "riba" in query.text
    assert query.sources_from == "bagaimana dengan hadits?"


def test_follow_up_without_source_keeps_previous_source() -> None:
    query = build_retrieval_query("yang kedua maksudnya apa?", "hadits tentang niat")

    assert query.follow_up
    assert query.sources_from == "hadits tentang niat"


def test_new_topic_is_not_a_follow_up() -> None:
    query = build_retrieval_query("ayat tentang sabar", "ayat tentang riba")

    assert not query.follow_up
    assert query.text == "ayat tentang sabar"


def test_no_previous_turn_means_no_follow_up() -> None:
    assert not build_retrieval_query("jelaskan ayat tadi", None).follow_up
    assert extract_keywords("terima kasih ya") == []


def test_last_turn() -> None:
    store = SessionStore(max_turns=5)
    sid = store.new_session_id()
    assert store.last_turn(sid) is None
    store.append(sid, "satu", "jawab satu")
    store.append(sid, "dua", "jawab dua")

    assert store.last_turn(sid) == ("dua", "jawab dua")


def test_format_context_variants() -> None:
    prev = [Passage("quran", "Al-Baqarah 2:278", "2:278", "teks lama")]
    new = [Passage("hadith", "Sahih al-Bukhari #1", "1", "teks baru")]

    both = format_context(new, previous=prev, has_history=True)
    assert both.index("DIBAHAS PADA JAWABAN SEBELUMNYA") < both.index("KONTEKS DALIL (hasil")
    assert "[Al-Baqarah 2:278] teks lama" in both and "[Sahih al-Bukhari #1] teks baru" in both
    assert "lanjutan" in format_context([], has_history=True)
    assert "tidak menemukan dalil yang cocok" in format_context([], has_history=False)


def test_pipeline_follow_up_rereads_previously_cited_dalil(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    FakeRunner = install_agents_stub(monkeypatch, [delta("ok")], final_output="ok")

    events = collect(
        runner.ask_streamed("jelaskan ayat tadi", session_id="s1", recent_turns=PREVIOUS)
    )

    prompt = FakeRunner.seen["prompt"]
    assert "DALIL YANG DIBAHAS PADA JAWABAN SEBELUMNYA" in prompt
    assert "[Al-Baqarah 2:278] Tinggalkan sisa riba" in prompt  # text re-read from the DB
    assert "[Sahih al-Bukhari #1] Sesungguhnya amal itu tergantung niat" in prompt
    assert ("tool", "local_db") in events  # retrieval also found "riba" via the previous topic


def test_pipeline_new_topic_with_history_skips_previous_dalil(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    FakeRunner = install_agents_stub(monkeypatch, [delta("ok")], final_output="ok")

    collect(runner.ask_streamed("hadits tentang niat", session_id="s1", recent_turns=PREVIOUS))

    assert "DIBAHAS PADA JAWABAN SEBELUMNYA" not in FakeRunner.seen["prompt"]


def test_pipeline_thanks_after_history_is_not_reported_as_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    FakeRunner = install_agents_stub(monkeypatch, [delta("sama-sama")], final_output="sama-sama")

    collect(runner.ask_streamed("terima kasih", session_id="s1", recent_turns=[("halo", "salam")]))

    assert "tidak menemukan dalil yang cocok" not in FakeRunner.seen["prompt"]
    assert "lanjutan" in FakeRunner.seen["prompt"]


def test_follow_up_sources_look_back_through_chained_follow_ups() -> None:
    from muslim_bot.agent import follow_up_sources

    turns = [
        ("ayat tentang riba", "Rujukan: Al-Baqarah 2:278."),
        ("jelaskan ayat tadi", "Penjelasan tanpa rujukan baru."),
    ]

    topic, cited = follow_up_sources(turns)

    assert topic == "ayat tentang riba"  # the latest question with a topic of its own
    assert cited == "Rujukan: Al-Baqarah 2:278."  # the latest answer that cited something
    assert follow_up_sources(None) == (None, None)
    assert follow_up_sources([("halo", "salam")]) == ("halo", None)


def test_recent_turns_oldest_first() -> None:
    store = SessionStore(max_turns=10)
    sid = store.new_session_id()
    for index in range(1, 6):
        store.append(sid, f"tanya {index}", f"jawab {index}")

    assert store.recent_turns(sid, limit=3) == [("tanya 3", "jawab 3"), ("tanya 4", "jawab 4"), ("tanya 5", "jawab 5")]
    assert store.recent_turns("belum-ada") == []


def test_pipeline_chained_follow_up_keeps_the_original_topic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    FakeRunner = install_agents_stub(monkeypatch, [delta("ok")], final_output="ok")
    turns = [*PREVIOUS, ("jelaskan ayat tadi", "Penjelasan tanpa rujukan baru.")]

    events = collect(runner.ask_streamed("bagaimana dengan hadits?", session_id="s1", recent_turns=turns))

    prompt = FakeRunner.seen["prompt"]
    assert ("tool", "local_db") in events
    assert "[Al-Baqarah 2:278]" in prompt  # re-read from the first answer, two turns back
