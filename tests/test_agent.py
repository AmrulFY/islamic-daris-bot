from __future__ import annotations

from types import SimpleNamespace

from muslim_bot.agent import extract_references, extract_used_tools


def test_extract_references_from_answer() -> None:
    answer = "Rujukan: Al-Baqarah 2:255 dan Sahih al-Bukhari #1."

    refs = extract_references(answer)

    assert [ref.type for ref in refs] == ["quran", "hadith"]
    assert refs[0].source_id == "2:255"
    assert refs[1].source_id == "1"


def test_extract_used_tools_is_deduplicated() -> None:
    result = SimpleNamespace(
        new_items=[
            SimpleNamespace(raw_item=SimpleNamespace(name="search_quran")),
            SimpleNamespace(raw_item=SimpleNamespace(tool_name="search_quran")),
            SimpleNamespace(raw_item=SimpleNamespace(name="search_hadith")),
        ]
    )

    assert extract_used_tools(result) == ["search_quran", "search_hadith"]
