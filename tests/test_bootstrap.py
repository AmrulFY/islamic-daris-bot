from __future__ import annotations

from pathlib import Path

import pytest

from muslim_bot.bootstrap import check_executable, check_workdir


def test_check_executable_passes_for_existing_command() -> None:
    check_executable("python", what="Test")


def test_check_executable_fails_with_helpful_message() -> None:
    with pytest.raises(RuntimeError, match="not found on PATH"):
        check_executable("definitely-not-a-real-command-xyz", what="Hadith seeder")


def test_check_workdir_passes_for_existing_directory(tmp_path: Path) -> None:
    check_workdir(tmp_path, what="Hadith seeder")


def test_check_workdir_fails_with_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        check_workdir(tmp_path / "no-such-dir", what="Hadith seeder")
