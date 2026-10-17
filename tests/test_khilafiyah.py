from __future__ import annotations

import pytest

from muslim_bot.agent import AGENT_INSTRUCTIONS, PIPELINE_INSTRUCTIONS
from tests.test_streaming import collect, delta, install_agents_stub, make_runner
from tests.test_retrieval import make_dbs


def rule_of(instructions: str) -> str:
    return next(line for line in instructions.splitlines() if "khilafiyah" in line)


@pytest.mark.parametrize("instructions", [PIPELINE_INSTRUCTIONS, AGENT_INSTRUCTIONS], ids=["pipeline", "agent loop"])
def test_both_prompts_tell_the_llm_how_to_handle_disputed_topics(instructions):
    rule = rule_of(instructions)

    assert "ulama berbeda pendapat" in rule  # say so first
    assert "jangan memihak" in rule and "pasti dianjurkan atau pasti dilarang" in rule
    assert "dalil yang sering dikutip" in rule  # not "the evidence that establishes it"
    assert "tidak menilai keabsahan hadits" in rule  # the database grades nothing
    assert "ulama atau ustaz yang dipercaya" in rule  # and point to a person, not to the bot


@pytest.mark.parametrize("instructions", [PIPELINE_INSTRUCTIONS, AGENT_INSTRUCTIONS], ids=["pipeline", "agent loop"])
def test_the_rule_is_limited_to_disputed_topics_and_not_to_date_questions(instructions):
    rule = rule_of(instructions)

    assert "Jangan menerapkan ini pada pertanyaan yang tidak diperselisihkan" in rule
    assert "pertanyaan yang hanya menanyakan tanggal" in rule  # "kapan Nisfu Syaban?" needs no sermon


@pytest.mark.parametrize("instructions", [PIPELINE_INSTRUCTIONS, AGENT_INSTRUCTIONS], ids=["pipeline", "agent loop"])
def test_the_rule_names_the_cases_that_prompted_it(instructions):
    rule = rule_of(instructions)
    assert "Nisfu Syaban" in rule and "Maulid" in rule and "Isra Miraj" in rule


def test_the_llm_actually_receives_the_rule(tmp_path, monkeypatch):
    """The text reaches the agent that writes the answer, not just a constant nobody reads."""
    make_dbs(tmp_path)
    runner = make_runner(tmp_path, monkeypatch)
    captured: dict = {}
    original = runner.build_agent

    def spy(runtime_config, mcp_servers, instructions=AGENT_INSTRUCTIONS):
        captured["instructions"] = instructions
        return original(runtime_config, mcp_servers, instructions)

    monkeypatch.setattr(runner, "build_agent", spy)
    install_agents_stub(monkeypatch, [delta("ok")], final_output="ok")

    collect(runner.ask_streamed("amalan malam nisfu syaban", session_id="s1"))

    assert "ulama berbeda pendapat" in captured["instructions"]
