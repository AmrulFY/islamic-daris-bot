from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Settings, load_settings
from .agent import SUBPROCESS_ENV_KEYS
from .db_status import hadith_db_ready, quran_db_ready, sqlite_count, sqlite_file_ok


def check_executable(command: str, *, what: str) -> None:
    if shutil.which(command):
        return
    raise RuntimeError(
        f"{what} command {command!r} was not found on PATH. "
        "Install Go from https://go.dev/dl/ (windows-amd64.msi on Windows), "
        "restart your terminal, or set the matching *_COMMAND env var "
        "to the full path of the executable."
    )


def check_workdir(cwd: Path, *, what: str) -> None:
    if cwd.is_dir():
        return
    raise RuntimeError(
        f"{what} working directory does not exist: {cwd}. "
        "Clone the sibling Go project so it sits next to the muslimbot folder, "
        "or set the matching *_CWD env var to the correct directory."
    )


def ensure_quran_db(settings: Settings, refresh: bool = False) -> None:
    source = settings.quran_source_db_path
    if not source.exists():
        raise FileNotFoundError(f"Quran database not found at {source}")

    settings.quran_db_path.parent.mkdir(parents=True, exist_ok=True)
    if refresh or not settings.quran_db_path.exists():
        shutil.copy2(source, settings.quran_db_path)

    if not quran_db_ready(settings.quran_db_path):
        count = sqlite_count(settings.quran_db_path, "ayahs")
        raise RuntimeError(f"Expected 6236 Quran ayahs, found {count}")


def _repair_if_needed(settings: Settings) -> None:
    """The Go seeder leaves chapters and the FTS index empty; see hadith_repair.

    Best effort: if the edition files can't be fetched, startup continues (the
    answer pipeline reads hadith_texts directly and does not need the repair).
    """
    from .hadith_repair import needs_repair, repair

    if not needs_repair(settings.hadith_db_path):
        return
    try:
        stats = repair(settings.hadith_db_path)
        print(f"Hadith DB repaired (chapters and search index rebuilt): {stats}")
    except Exception as exc:  # noqa: BLE001 - never block startup on this.
        print(f"WARNING: could not repair Hadith DB chapters/search index: {exc}")


def ensure_hadith_db(settings: Settings, seed: bool = False) -> None:
    settings.hadith_db_path.parent.mkdir(parents=True, exist_ok=True)
    if hadith_db_ready(settings.hadith_db_path) and not seed:
        _repair_if_needed(settings)
        return

    if settings.hadith_db_path.exists() and (seed or not sqlite_file_ok(settings.hadith_db_path)):
        settings.hadith_db_path.unlink()

    check_executable(settings.hadith_seeder_command, what="Hadith seeder")
    check_workdir(settings.hadith_seeder_cwd, what="Hadith seeder")

    env = {key: os.environ[key] for key in SUBPROCESS_ENV_KEYS if key in os.environ}
    env.update({
        "DATABASE_PATH": str(settings.hadith_db_path),
        "LOG_LEVEL": "silent",
    })
    subprocess.run(
        [settings.hadith_seeder_command, *settings.hadith_seeder_args],
        cwd=settings.hadith_seeder_cwd,
        env=env,
        check=True,
    )

    if not hadith_db_ready(settings.hadith_db_path):
        count = sqlite_count(settings.hadith_db_path, "hadiths")
        raise RuntimeError(f"Hadith database is incomplete after seed; found {count} hadith rows")
    _repair_if_needed(settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare local Quran and Hadith databases.")
    parser.add_argument("--refresh-quran", action="store_true", help="Copy Quran DB even if it already exists.")
    parser.add_argument("--seed-hadith", action="store_true", help="Recreate and seed the Hadith DB.")
    parser.add_argument("--skip-hadith", action="store_true", help="Only prepare Quran DB.")
    args = parser.parse_args(argv)

    settings = load_settings()
    ensure_quran_db(settings, refresh=args.refresh_quran)
    print(f"Quran DB ready: {settings.quran_db_path}")

    if not args.skip_hadith:
        ensure_hadith_db(settings, seed=args.seed_hadith)
        print(f"Hadith DB ready: {settings.hadith_db_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
