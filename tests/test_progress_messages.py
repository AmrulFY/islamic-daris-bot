from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

import muslim_bot.telegram_bot as tg_bot
from muslim_bot.progress_messages import PROGRESS_POOLS, classify_progress, pick_progress
from tests.test_telegram_bot import ProgressFakeMessage


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("ayat tentang sabar", "quran"),
        ("surah Al-Mulk", "quran"),
        ("hadis tentang niat", "hadith"),
        ("hadits riwayat Bukhari tentang puasa", "hadith"),
        ("ayat dan hadis tentang sabar", "dalil"),
        ("ayat kursi", "lookup"),
        ("2:255", "lookup"),
        ("Sahih al-Bukhari #1", "lookup"),
        ("ayat acak dong", "random"),
        ("bagaimana hukum jual beli online", "ruling"),
        ("apa itu zakat fitrah", "concept"),
        ("jelaskan perbedaan sunnah dan wajib", "ruling"),
        ("Assalamu'alaikum", "greeting"),
        ("halo kak", "greeting"),
        ("selamat pagi", "greeting"),
        ("terima kasih banyak ya", "thanks"),
        ("jazakallahu khairan", "thanks"),
        ("tolong rekomendasi bacaan sebelum tidur", "general"),
    ],
)
def test_classify_progress(text: str, category: str) -> None:
    assert classify_progress(text) == category


def test_greeting_with_real_question_is_not_silent() -> None:
    assert classify_progress("Halo, ayat tentang sabar") == "quran"
    assert classify_progress("terima kasih, lalu hadis tentang niat?") == "hadith"


def test_silent_categories_have_no_text() -> None:
    assert pick_progress("greeting") == (None, None)
    assert pick_progress("thanks") == (None, None)


def test_pick_progress_never_repeats_last_index() -> None:
    rng = random.Random(0)
    for category, pool in PROGRESS_POOLS.items():
        last = None
        for _ in range(50):
            text, index = pick_progress(category, last, rng)
            assert text == pool[index]
            assert index != last
            last = index


def test_general_text_never_mentions_dalil() -> None:
    assert not any("dalil" in text.lower() for text in PROGRESS_POOLS["general"])


def _context(chat_data=None):
    return SimpleNamespace(
        bot_data={"settings": SimpleNamespace(telegram_api_base="http://x")},
        chat_data={} if chat_data is None else chat_data,
    )


def _post_after(delay: float, answer: str = "Jawaban."):
    async def post(*args, **kwargs):
        await asyncio.sleep(delay)
        return {"answer": answer, "references": [], "warnings": []}

    return post


def test_progress_text_matches_topic(monkeypatch) -> None:
    monkeypatch.setattr(tg_bot, "post_chat", _post_after(0.1))
    monkeypatch.setattr(tg_bot, "PROGRESS_DELAY_SECONDS", 0.01)
    message = ProgressFakeMessage("apa itu zakat fitrah")

    asyncio.run(tg_bot.text_handler(SimpleNamespace(message=message), _context()))

    assert message.progress.text in PROGRESS_POOLS["concept"]
    assert message.progress.deleted is True


def test_instant_answer_shows_no_progress(monkeypatch) -> None:
    monkeypatch.setattr(tg_bot, "post_chat", _post_after(0))
    monkeypatch.setattr(tg_bot, "PROGRESS_DELAY_SECONDS", 0.5)
    message = ProgressFakeMessage("hadis tentang niat")

    asyncio.run(tg_bot.text_handler(SimpleNamespace(message=message), _context()))

    assert message.progress is None
    assert any(call.text == "Jawaban." for call in message.calls)


def test_greeting_gets_no_progress_even_when_slow(monkeypatch) -> None:
    monkeypatch.setattr(tg_bot, "post_chat", _post_after(0.1, "Wa'alaikumussalam."))
    monkeypatch.setattr(tg_bot, "PROGRESS_DELAY_SECONDS", 0.01)
    message = ProgressFakeMessage("Assalamu'alaikum")

    asyncio.run(tg_bot.text_handler(SimpleNamespace(message=message), _context()))

    assert message.progress is None


def test_progress_is_cleaned_up_after_backend_error(monkeypatch) -> None:
    async def failing_post(*args, **kwargs):
        await asyncio.sleep(0.1)
        raise RuntimeError("boom")

    monkeypatch.setattr(tg_bot, "post_chat", failing_post)
    monkeypatch.setattr(tg_bot, "PROGRESS_DELAY_SECONDS", 0.01)
    message = ProgressFakeMessage("ayat tentang sabar")

    asyncio.run(tg_bot.text_handler(SimpleNamespace(message=message), _context()))

    assert message.progress.deleted is True
    assert any(call.text == tg_bot.ERROR_REPLY for call in message.calls)


def test_consecutive_progress_texts_differ_within_a_chat() -> None:
    chat_data: dict = {}

    texts = [tg_bot.choose_progress_text("hadis tentang niat", chat_data) for _ in range(30)]

    assert all(a != b for a, b in zip(texts, texts[1:]))
