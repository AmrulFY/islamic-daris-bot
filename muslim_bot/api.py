from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, FastAPI, Request
from openai import APIConnectionError, APIStatusError, APITimeoutError
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import AgentLoopLimitError, AgentReply, MuslimBotRunner, list_mcp_tools
from .answer_cache import AnswerCache
from .config import PROJECT_ROOT, Settings, load_settings
from .db_status import hadith_db_ready, quran_db_ready
from .rate_limit import RateLimiter
from .runtime_config import RuntimeConfig, RuntimeConfigStore
from .schemas import (
    AdminConfigResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryClearResponse,
    MCPTestResponse,
    PrefsResponse,
    PrefsUpdate,
)
from .semantic_index import build as build_index, index_complete
from .session_store import SessionStore, UserPrefs


def configure_logging() -> None:
    """Ensure our package logs are visible without touching uvicorn's setup.

    Uvicorn only configures its own loggers, leaving the root logger at
    WARNING — so logger.info calls would otherwise be silently swallowed.
    Attaching a handler to our namespace (with propagation off) avoids
    duplicating uvicorn's own log lines.
    """
    package_logger = logging.getLogger("muslim_bot")
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        package_logger.addHandler(handler)
    package_logger.propagate = False
    package_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


configure_logging()


@lru_cache(maxsize=1)
def get_answer_cache() -> AnswerCache:
    settings = get_settings()
    return AnswerCache(
        settings.session_db_path,
        ttl_days=settings.answer_cache_ttl_days,
        enabled=settings.chat_cache_enabled,
    )


_semantic_stop = threading.Event()


async def _semantic_startup() -> None:
    """Build the semantic index if it is missing, then warm up the embedding client.

    Runs in the background so the API serves (keyword search only) while a first
    boot embeds the corpus, which takes several minutes. The build resumes where
    it stopped if the process restarts.
    """
    runner = get_runner()
    settings = get_settings()
    if runner.semantic is None:
        return
    llm = runner.load_runtime_config().llm
    if not llm.is_configured:
        return
    if settings.semantic_auto_index and not index_complete(settings.semantic_db_path):
        if quran_db_ready(settings.quran_db_path) and hadith_db_ready(settings.hadith_db_path):
            logging.getLogger(__name__).info("semantic index incomplete; building in the background")
            runner.semantic.building = True
            try:
                await asyncio.to_thread(build_index, ["quran", "hadith"], _semantic_stop.is_set)
                runner.semantic.reload()
            except Exception:  # noqa: BLE001 - keyword search keeps working.
                logging.getLogger(__name__).exception("semantic index build failed; will retry on next start")
            finally:
                runner.semantic.building = False
        else:
            logging.getLogger(__name__).warning("semantic index not built: Quran/Hadith databases are not ready")
    await runner.semantic.warm_up(api_key=llm.api_key, base_url=llm.base_url)


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_session_store().purge_inactive(get_settings().session_retention_days)
    get_answer_cache().purge_expired()
    semantic_startup = asyncio.create_task(_semantic_startup())
    yield
    _semantic_stop.set()
    semantic_startup.cancel()
    await get_runner().mcp_manager.aclose()
    get_session_store().close()


app = FastAPI(title="Islamic Daris Bot", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "muslim_bot" / "static"), name="static")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    return SessionStore(
        max_turns=get_settings().session_max_turns,
        db_path=get_settings().session_db_path,
    )


@lru_cache(maxsize=1)
def get_runner() -> MuslimBotRunner:
    return MuslimBotRunner(settings=get_settings())


@lru_cache(maxsize=1)
def get_config_store() -> RuntimeConfigStore:
    return RuntimeConfigStore(settings=get_settings())


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().chat_rate_limit_per_minute)


def check_rate_limit(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        key = forwarded_for.split(",")[0].strip() or "unknown"
    elif request.client:
        key = request.client.host
    else:
        key = "unknown"
    limiter.check(key)


def require_admin(
    x_admin_token: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.admin_token:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN is not configured")
    if x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid admin token")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "muslim_bot" / "static" / "index.html")


@app.get("/health", response_model=HealthResponse)
def health(
    settings: Settings = Depends(get_settings),
    store: RuntimeConfigStore = Depends(get_config_store),
) -> HealthResponse:
    runtime_config = store.load()
    quran_ready = quran_db_ready(settings.quran_db_path)
    hadith_ready = hadith_db_ready(settings.hadith_db_path)
    status = "ok" if runtime_config.llm.is_configured and quran_ready and hadith_ready else "degraded"
    searcher = get_runner().semantic
    return HealthResponse(
        status=status,
        llm_configured=runtime_config.llm.is_configured,
        quran_db_ready=quran_ready,
        hadith_db_ready=hadith_ready,
        quran_api_dir=str(settings.quran_api_dir),
        hadith_api_dir=str(settings.hadith_api_dir),
        runtime_config_path=str(settings.runtime_config_path),
        enabled_mcp_servers=sum(1 for server in runtime_config.mcp_servers if server.enabled),
        # Informational only: a missing or still-building index never fails the health check.
        semantic_index=searcher.status() if searcher is not None else "disabled",
    )


@app.get("/api/config", response_model=AdminConfigResponse, dependencies=[Depends(require_admin)])
def get_config(store: RuntimeConfigStore = Depends(get_config_store)) -> AdminConfigResponse:
    return AdminConfigResponse(config=store.load_redacted(), saved=store.path.exists())


@app.put("/api/config", response_model=AdminConfigResponse, dependencies=[Depends(require_admin)])
def put_config(
    config: RuntimeConfig,
    store: RuntimeConfigStore = Depends(get_config_store),
) -> AdminConfigResponse:
    saved = store.save_preserving_masked_secrets(config)
    return AdminConfigResponse(config=store.load_redacted(), saved=store.path.exists())


@app.post("/api/config/reset", response_model=AdminConfigResponse, dependencies=[Depends(require_admin)])
def reset_config(store: RuntimeConfigStore = Depends(get_config_store)) -> AdminConfigResponse:
    store.reset()
    return AdminConfigResponse(config=store.load_redacted(), saved=True)


@app.post("/api/mcp/test", response_model=MCPTestResponse, dependencies=[Depends(require_admin)])
async def test_mcp(store: RuntimeConfigStore = Depends(get_config_store)) -> MCPTestResponse:
    return MCPTestResponse(servers=await list_mcp_tools(store.load()))


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_session_store),
    runner: MuslimBotRunner = Depends(get_runner),
    cache: AnswerCache = Depends(get_answer_cache),
    _: None = Depends(check_rate_limit),
) -> ChatResponse:
    runtime_config = RuntimeConfigStore(settings).load()
    if not runtime_config.llm.is_configured:
        raise HTTPException(status_code=503, detail="LLM connection is not configured")

    session_id = store.ensure_session(request.session_id)
    prefs = store.get_prefs(session_id)
    language = prefs.language or request.language
    history_prompt = store.history_prompt(session_id, request.message)
    has_history = history_prompt != request.message
    use_cache = not has_history and not prefs.customizes_answer
    history_prompt = _with_prefs(prefs, history_prompt)
    recent_turns = store.recent_turns(session_id, limit=_recent_limit(prefs))

    fast = runner.try_fast_path(request.message, language=language)
    if fast is not None:
        store.append(session_id, request.message, fast.answer)
        return _reply_to_response(session_id, fast)

    if use_cache:
        cached = cache.get(request.message, language)
        if cached is not None:
            store.append(session_id, request.message, cached["answer"])
            return _cached_to_response(session_id, cached)

    try:
        reply = await runner.ask(
            request.message,
            session_id=session_id,
            history_prompt=history_prompt,
            language=language,
            recent_turns=recent_turns,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Agent request timed out") from exc
    except AgentLoopLimitError as exc:
        raise HTTPException(status_code=508, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (APIConnectionError, APIStatusError) as exc:
        status, detail = llm_error_response(exc) or (502, "LLM gateway error")
        raise HTTPException(status_code=status, detail=detail) from exc

    store.append(session_id, request.message, reply.answer)
    if use_cache:
        cache.put(
            request.message,
            language,
            reply.answer,
            {
                "references": [ref.model_dump() for ref in reply.references],
                "used_tools": reply.used_tools,
                "warnings": reply.warnings,
            },
        )
    return _reply_to_response(session_id, reply)


def _with_prefs(prefs: UserPrefs, history_prompt: str) -> str:
    block = prefs.prompt_block()
    return f"{block}\n\n{history_prompt}" if block else history_prompt


def _recent_limit(prefs: UserPrefs) -> int:
    """Follow-up detection must not see turns the user asked the bot not to keep."""
    default = 4
    return default if prefs.history_turns is None else min(default, prefs.history_turns)


def _prefs_response(session_id: str, prefs: UserPrefs, store: SessionStore) -> PrefsResponse:
    return PrefsResponse(
        session_id=session_id,
        language=prefs.language,
        style=prefs.style,
        note=prefs.note,
        history_turns=prefs.history_turns,
        max_history_turns=store.max_turns,
    )


@app.get("/sessions/{session_id}/prefs", response_model=PrefsResponse)
def get_prefs(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
    _: None = Depends(check_rate_limit),
) -> PrefsResponse:
    return _prefs_response(session_id, store.get_prefs(session_id), store)


@app.put("/sessions/{session_id}/prefs", response_model=PrefsResponse)
def put_prefs(
    session_id: str,
    update: PrefsUpdate,
    store: SessionStore = Depends(get_session_store),
    _: None = Depends(check_rate_limit),
) -> PrefsResponse:
    changes = {name: getattr(update, name) for name in update.model_fields_set}
    try:
        prefs = store.set_prefs(session_id, **changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _prefs_response(session_id, prefs, store)


@app.delete("/sessions/{session_id}/prefs", response_model=PrefsResponse)
def delete_prefs(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
    _: None = Depends(check_rate_limit),
) -> PrefsResponse:
    store.clear_prefs(session_id)
    return _prefs_response(session_id, UserPrefs(), store)


@app.delete("/sessions/{session_id}/history", response_model=HistoryClearResponse)
def delete_history(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
    _: None = Depends(check_rate_limit),
) -> HistoryClearResponse:
    return HistoryClearResponse(session_id=session_id, deleted_turns=store.clear_history(session_id))


def _reply_to_response(session_id: str, reply: AgentReply) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        answer=reply.answer,
        references=reply.references,
        used_tools=reply.used_tools,
        warnings=reply.warnings,
    )


def _cached_to_response(session_id: str, cached: dict) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        answer=cached["answer"],
        references=cached["references"],
        used_tools=cached["used_tools"],
        warnings=cached["warnings"],
    )


def llm_error_response(exc: Exception) -> tuple[int, str] | None:
    """Map a failed LLM gateway call to (HTTP status, message), None if not one."""
    if isinstance(exc, APITimeoutError):
        return 504, "LLM request timed out"
    if isinstance(exc, APIConnectionError):
        return 503, "LLM gateway unreachable"
    if isinstance(exc, APIStatusError):
        return 502, f"LLM gateway error ({exc.status_code})"
    return None


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    store: SessionStore = Depends(get_session_store),
    runner: MuslimBotRunner = Depends(get_runner),
    cache: AnswerCache = Depends(get_answer_cache),
    _: None = Depends(check_rate_limit),
) -> StreamingResponse:
    runtime_config = RuntimeConfigStore(settings).load()
    if not runtime_config.llm.is_configured:
        raise HTTPException(status_code=503, detail="LLM connection is not configured")

    session_id = store.ensure_session(request.session_id)
    prefs = store.get_prefs(session_id)
    language = prefs.language or request.language
    history_prompt = store.history_prompt(session_id, request.message)
    has_history = history_prompt != request.message
    use_cache = not has_history and not prefs.customizes_answer
    history_prompt = _with_prefs(prefs, history_prompt)
    recent_turns = store.recent_turns(session_id, limit=_recent_limit(prefs))
    idle_timeout = runtime_config.agent_timeout_seconds

    async def instant(reply: AgentReply):
        store.append(session_id, request.message, reply.answer)
        yield _sse("delta", {"text": reply.answer})
        yield _sse("done", _reply_to_response(session_id, reply).model_dump())

    async def generate():
        fast = runner.try_fast_path(request.message, language=language)
        if fast is not None:
            async for event in instant(fast):
                yield event
            return

        if use_cache:
            cached = cache.get(request.message, language)
            if cached is not None:
                async for event in instant(
                    AgentReply(
                        answer=cached["answer"],
                        references=cached["references"],
                        used_tools=cached["used_tools"],
                        warnings=cached["warnings"],
                    )
                ):
                    yield event
                return

        gen = runner.ask_streamed(
            request.message,
            session_id=session_id,
            history_prompt=history_prompt,
            language=language,
            recent_turns=recent_turns,
        )
        completed_answer = ""
        completed_reply: AgentReply | None = None
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(gen.__anext__(), timeout=idle_timeout)
                except StopAsyncIteration:
                    break
                if kind == "delta":
                    yield _sse("delta", {"text": payload})
                elif kind == "tool":
                    yield _sse("tool", {"name": payload})
                elif kind == "done":
                    completed_answer = payload.answer
                    completed_reply = payload
                    yield _sse("done", _reply_to_response(session_id, payload).model_dump())
        except asyncio.TimeoutError:
            yield _sse("error", {"detail": "Agent request timed out"})
        except AgentLoopLimitError as exc:
            yield _sse("error", {"detail": str(exc)})
        except RuntimeError as exc:
            yield _sse("error", {"detail": str(exc)})
        except (APIConnectionError, APIStatusError) as exc:
            yield _sse("error", {"detail": (llm_error_response(exc) or (502, "LLM gateway error"))[1]})
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass
        if completed_answer and completed_reply and use_cache:
            store.append(session_id, request.message, completed_answer)
            cache.put(
                request.message,
                language,
                completed_answer,
                {
                    "references": [ref.model_dump() for ref in completed_reply.references],
                    "used_tools": completed_reply.used_tools,
                    "warnings": completed_reply.warnings,
                },
            )
        elif completed_answer:
            store.append(session_id, request.message, completed_answer)

    return StreamingResponse(generate(), media_type="text/event-stream")
