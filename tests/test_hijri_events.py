from __future__ import annotations

from datetime import date

import httpx
import pytest

from muslim_bot import hijri
from muslim_bot.hijri import SOURCE_COMPUTED, SOURCE_FALLBACK, SOURCE_TABLE, parse_api_response
from tests.test_hijri import TABLE_PAYLOAD, TODAY, chat_client, make_calendar, run  # noqa: F401 (fixture)

EVENTS_PAYLOAD_2027 = {
    "events": [
        {"event": "awal_ramadan", "name": "Awal Ramadan", "hijri": "1448-09-01", "gregorian": "2027-02-08", "source": "mabims-computed"},
        {"event": "idul_fitri", "name": "Idul Fitri", "hijri": "1448-10-01", "gregorian": "2027-03-10", "source": "mabims-computed"},
        {"event": "idul_adha", "name": "Idul Adha", "hijri": "1448-12-10", "gregorian": "2027-05-17", "source": "mabims-computed"},
        {"event": "1_muharram", "name": "Tahun Baru Islam", "hijri": "1449-01-01", "gregorian": "2027-06-06", "source": "mabims-computed"},
        {"event": "maulid_nabi", "name": "Maulid Nabi Muhammad Shallallahu Alaihi Wasallam", "hijri": "1449-03-12", "gregorian": "2027-08-15", "source": "mabims-computed"},
    ],
    "count": 5,
}


def events_2027():
    return hijri.with_derived_events(hijri.parse_events_response(EVENTS_PAYLOAD_2027))


def by_key(events):
    return {event.key: event for event in events}


# --- holiday data ---------------------------------------------------------------------


def test_parse_events_response_and_derived_fasts():
    events = events_2027()
    found = by_key(events)
    assert found["awal_ramadan"].day == date(2027, 2, 8)
    assert found["maulid_nabi"].name == "Maulid Nabi Muhammad SAW"  # our short name, not the API's
    ramadan = found["puasa_ramadan"]
    assert (ramadan.day, ramadan.last_day) == (date(2027, 2, 8), date(2027, 3, 9))  # until the day before Idul Fitri
    assert found["puasa_arafah"].day == date(2027, 5, 16)
    assert found["puasa_tasua"].day == date(2027, 6, 14)
    assert found["puasa_asyura"].day == date(2027, 6, 15)
    assert [event.day for event in events] == sorted(event.day for event in events)


def test_parse_events_response_rejects_malformed_payload():
    with pytest.raises(ValueError):
        hijri.parse_events_response({"events": [{"event": "idul_fitri"}]})
    with pytest.raises(ValueError):
        hijri.parse_events_response({})


def test_calculate_events_local_fallback():
    found = by_key(hijri.calculate_events(2026))
    assert found["idul_adha"].day == date(2026, 5, 27)
    assert found["awal_ramadan"].day.year == 2026
    assert all(event.source == SOURCE_FALLBACK for event in found.values())
    assert hijri.calculate_events(2200) == []


def test_events_use_api_and_cache():
    calendar, requests = make_calendar(lambda request: httpx.Response(200, json=EVENTS_PAYLOAD_2027))
    first = run(calendar.events(2027))
    second = run(calendar.events(2027))
    assert first == second
    assert by_key(first)["idul_adha"].source == SOURCE_COMPUTED
    assert len(requests) == 1
    assert requests[0].url.params["year"] == "2027"
    assert requests[0].url.params["calendar"] == "gregorian"


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500), httpx.Response(200, json={"events": [], "count": 0}), httpx.Response(200, text="oops")],
)
def test_events_fall_back_when_api_fails_or_is_empty(response):
    calendar, _ = make_calendar(lambda request: response)
    events = run(calendar.events(2027))
    assert events and all(event.source == SOURCE_FALLBACK for event in events)
    assert "puasa_ramadan" in by_key(events)


# --- the calendar block the LLM reads ------------------------------------------------


def test_calendar_context_marks_past_and_upcoming_events():
    past = hijri.HijriEvent("idul_fitri", "Idul Fitri", date(2026, 3, 21), SOURCE_TABLE, "1447-10-01")
    text = hijri.format_calendar_context(TODAY, parse_api_response(TABLE_PAYLOAD), [past, *events_2027()])

    assert "Hari ini: Sabtu, 17 Oktober 2026 = 6 Jumadil Awal 1448 H [tabel kalender Kemenag]" in text
    assert "Idul Fitri: Sabtu, 21 Maret 2026 (1 Syawal 1447 H) [sudah lewat; tabel kalender Kemenag]" in text
    assert (
        "Awal Ramadan: Senin, 8 Februari 2027 (1 Ramadhan 1448 H) [akan datang, 114 hari lagi; "
        "perkiraan hitungan MABIMS; penetapan resmi oleh pemerintah melalui sidang isbat]"
    ) in text
    assert "Puasa Ramadan: Senin, 8 Februari 2027 sampai Selasa, 9 Maret 2027" in text
    assert "sudah lewat; tabel kalender Kemenag; penetapan" not in text  # past events need no isbat note
    assert "Hari besar dan puasa tahun 2026:" in text and "Hari besar dan puasa tahun 2027:" in text


def test_calendar_context_ongoing_ramadan():
    text = hijri.format_calendar_context(date(2027, 2, 20), None, events_2027())
    assert "(tanggal Hijriyah tidak tersedia)" in text
    assert "Puasa Ramadan: Senin, 8 Februari 2027 sampai Selasa, 9 Maret 2027 [sedang berlangsung" in text


@pytest.mark.parametrize(
    "message",
    [
        "kapan puasa di tahun ini?",
        "kapan idul adha?",
        "kapan lebaran",
        "tanggal berapa 1 ramadan",
        "kapan 1 muharram tahun depan",
        "apa itu kalender hijriyah",
        "tahun baru islam",
        "berapa hari lagi puasa arafah",
    ],
)
def test_calendar_related_messages(message):
    assert hijri.is_calendar_related(message)


@pytest.mark.parametrize(
    "message",
    [
        "hadits tentang keutamaan puasa",
        "ayat tentang puasa",
        "doa buka puasa",
        "jelaskan tentang ramadan",
        "hadits tentang niat",
        "terima kasih",
    ],
)
def test_ordinary_dalil_questions_are_not_calendar_related(message):
    assert not hijri.is_calendar_related(message)


def test_follow_up_to_calendar_question_keeps_calendar_context():
    assert hijri.wants_calendar("kalau tahun depan?", "kapan puasa tahun ini?")
    assert hijri.wants_calendar("kalau idul fitri?", "kapan puasa tahun ini?")
    assert not hijri.wants_calendar("jelaskan ayat tadi", "hadits tentang sabar")
    assert not hijri.wants_calendar("tolong jelaskan " + "panjang " * 20, "kapan puasa tahun ini?")


class StubCalendarSource:
    """Stands in for a HijriCalendar (no network)."""

    def __init__(self):
        self.calls = 0

    async def convert(self, day=None):
        self.calls += 1
        return parse_api_response(TABLE_PAYLOAD)

    async def events(self, year):
        self.calls += 1
        return events_2027() if year == 2027 else []


def test_build_calendar_context_only_for_calendar_questions(monkeypatch):
    monkeypatch.setattr(hijri, "today_in", lambda tz_name=hijri.DEFAULT_TIMEZONE: TODAY)
    calendar = StubCalendarSource()

    assert run(hijri.build_calendar_context(calendar, "hadits tentang niat")) is None
    assert calendar.calls == 0

    text = run(hijri.build_calendar_context(calendar, "kapan puasa di tahun ini?"))
    assert text.startswith("KONTEKS KALENDER")
    assert "8 Februari 2027" in text


def test_build_calendar_context_survives_calendar_errors():
    class Broken:
        async def convert(self, day=None):
            raise RuntimeError("boom")

        async def events(self, year):
            return []

    assert run(hijri.build_calendar_context(Broken(), "kapan lebaran")) is None


# --- through the pipeline: the LLM answers, conversationally -------------------------


def pipeline_runner(tmp_path, monkeypatch):
    from tests.test_retrieval import make_dbs
    from tests.test_streaming import delta, install_agents_stub, make_runner

    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    runner.hijri = StubCalendarSource()
    monkeypatch.setattr(hijri, "today_in", lambda tz_name=hijri.DEFAULT_TIMEZONE: TODAY)
    fake = install_agents_stub(monkeypatch, [delta("ok")], final_output="ok")
    return runner, fake


def test_pipeline_gives_the_llm_the_calendar_for_calendar_questions(tmp_path, monkeypatch):
    from tests.test_streaming import collect

    runner, fake = pipeline_runner(tmp_path, monkeypatch)

    events = collect(runner.ask_streamed("kapan puasa di tahun ini?", session_id="s1"))

    prompt = fake.seen["prompt"]
    assert "KONTEKS KALENDER" in prompt
    assert "Awal Ramadan: Senin, 8 Februari 2027" in prompt
    assert ("tool", "hijri_calendar") in events
    assert events[-1][1].used_tools[-1] == "hijri_calendar"


def test_pipeline_skips_calendar_for_ordinary_questions(tmp_path, monkeypatch):
    from tests.test_streaming import collect

    runner, fake = pipeline_runner(tmp_path, monkeypatch)

    events = collect(runner.ask_streamed("hadits tentang niat", session_id="s1"))

    assert "KONTEKS KALENDER" not in fake.seen["prompt"]
    assert ("tool", "hijri_calendar") not in events


def test_pipeline_calendar_follow_up_in_conversation(tmp_path, monkeypatch):
    from tests.test_streaming import collect

    runner, fake = pipeline_runner(tmp_path, monkeypatch)
    previous = [("kapan puasa di tahun ini?", "Ramadan tahun ini sudah lewat; berikutnya 8 Februari 2027.")]

    collect(runner.ask_streamed("kalau idul fitri kapan?", session_id="s1", recent_turns=previous))
    assert "KONTEKS KALENDER" in fake.seen["prompt"]

    collect(runner.ask_streamed("terima kasih", session_id="s1", recent_turns=previous))
    assert "KONTEKS KALENDER" in fake.seen["prompt"]  # a short follow-up keeps the calendar

    collect(runner.ask_streamed("hadits tentang niat", session_id="s1", recent_turns=[("halo", "salam")]))
    assert "KONTEKS KALENDER" not in fake.seen["prompt"]


def test_pipeline_instructions_tell_the_llm_to_use_only_the_calendar_block():
    from muslim_bot.agent import PIPELINE_INSTRUCTIONS

    assert "KONTEKS KALENDER" in PIPELINE_INSTRUCTIONS
    assert "kalender Hijriah, kalkulator" not in PIPELINE_INSTRUCTIONS  # no longer listed as unavailable


def test_calendar_answers_are_never_served_from_or_stored_in_the_answer_cache(chat_client):  # noqa: F811
    from tests.test_api import FakeRunner

    client, _api, monkeypatch = chat_client
    calls: list[str] = []
    original = FakeRunner.ask

    async def counting(self, message, **kwargs):
        calls.append(message)
        return await original(self, message, **kwargs)

    monkeypatch.setattr(FakeRunner, "ask", counting)

    for session in ("a", "b", "c", "d"):
        client.post("/chat", json={"message": "kapan puasa di tahun ini?", "session_id": session})
    assert len(calls) == 4  # asked every time, never cached

    for session in ("e", "f"):
        client.post("/chat", json={"message": "hadis tentang niat", "session_id": session})
    assert len(calls) == 5  # ordinary questions are still cached


def test_chain_of_follow_ups_keeps_the_calendar():
    chain = ["kapan puasa tahun ini?", "kalau tahun depan?", "dan idul fitri?"]
    assert hijri.wants_calendar("terima kasih", chain)
    assert hijri.wants_calendar("terima kasih", chain[0])  # a single string still works
    assert not hijri.wants_calendar("terima kasih", ["hadits tentang niat", "ayat tentang sabar"])
    assert not hijri.wants_calendar("terima kasih", None)


@pytest.mark.parametrize("message", ["kapan Isra Miraj?", "kapan nisfu syaban", "kapan 1 rajab"])
def test_other_islamic_dates_still_get_the_calendar_block(message):
    # The data has no such holiday, so the LLM must say so rather than guess: it needs the block to know.
    assert hijri.is_calendar_related(message)


def test_unrelated_words_do_not_trigger_the_calendar():
    assert not hijri.is_calendar_related("kapan israel menyerang gaza")


def test_pipeline_follow_up_chain_reaches_back_three_turns(tmp_path, monkeypatch):
    from tests.test_streaming import collect

    runner, fake = pipeline_runner(tmp_path, monkeypatch)
    turns = [
        ("kapan puasa di tahun ini?", "Jawaban satu."),
        ("kalau tahun depan?", "Jawaban dua."),
        ("terima kasih", "Sama-sama."),
    ]

    collect(runner.ask_streamed("oke jelas", session_id="s1", recent_turns=turns))

    assert "KONTEKS KALENDER" in fake.seen["prompt"]


def test_new_dalil_topic_after_calendar_questions_drops_the_calendar():
    chain = ["kapan puasa tahun ini?", "kapan idul adha tahun depan?"]
    assert not hijri.wants_calendar("ayat tentang sabar", chain)
    assert not hijri.wants_calendar("hadits tentang niat?", chain)
    assert hijri.wants_calendar("kalau idul fitri?", chain)
    assert hijri.is_calendar_related("kapan puasa arafah")  # a calendar question that names puasa still counts
