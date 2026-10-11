from __future__ import annotations

import argparse
import asyncio
import sys

from .agent import MuslimBotRunner
from .config import load_settings
from .runtime_config import RuntimeConfigStore
from .session_store import SessionStore


async def _ask_once(message: str, session_id: str | None = None) -> int:
    settings = load_settings()
    if not settings.has_openai_api_key:
        runtime_config = RuntimeConfigStore(settings).load()
        if not runtime_config.llm.is_configured:
            print("LLM connection is not configured. Add it in .env or the admin UI first.", file=sys.stderr)
            return 2
    else:
        runtime_config = RuntimeConfigStore(settings).load()

    store = SessionStore(
        max_turns=runtime_config.session_max_turns,
        db_path=settings.session_db_path,
    )
    active_session_id = store.ensure_session(session_id)
    runner = MuslimBotRunner(settings=settings)
    reply = await runner.ask(
        message,
        session_id=active_session_id,
        history_prompt=store.history_prompt(active_session_id, message),
    )
    print(reply.answer)
    return 0


async def _interactive() -> int:
    settings = load_settings()
    if not settings.has_openai_api_key:
        runtime_config = RuntimeConfigStore(settings).load()
        if not runtime_config.llm.is_configured:
            print("LLM connection is not configured. Add it in .env or the admin UI first.", file=sys.stderr)
            return 2
    else:
        runtime_config = RuntimeConfigStore(settings).load()

    store = SessionStore(
        max_turns=runtime_config.session_max_turns,
        db_path=settings.session_db_path,
    )
    session_id = store.new_session_id()
    runner = MuslimBotRunner(settings=settings)

    print("Islamic Daris Bot siap. Ketik 'exit' untuk keluar.")
    while True:
        try:
            message = input("> ").strip()
        except EOFError:
            print()
            return 0
        if message.lower() in {"exit", "quit"}:
            return 0
        if not message:
            continue
        reply = await runner.ask(
            message,
            session_id=session_id,
            history_prompt=store.history_prompt(session_id, message),
        )
        store.append(session_id, message, reply.answer)
        print(reply.answer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with Islamic Daris Bot.")
    parser.add_argument("--once", help="Ask one question and exit.")
    parser.add_argument("--session-id", help="Optional session id for --once.")
    args = parser.parse_args(argv)

    if args.once:
        return asyncio.run(_ask_once(args.once, session_id=args.session_id))
    return asyncio.run(_interactive())


if __name__ == "__main__":
    sys.exit(main())
