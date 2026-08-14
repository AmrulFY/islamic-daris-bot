from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import MuslimBotRunner, list_mcp_tools
from .config import PROJECT_ROOT, Settings, load_settings
from .db_status import hadith_db_ready, quran_db_ready
from .runtime_config import RuntimeConfig, RuntimeConfigStore
from .schemas import AdminConfigResponse, ChatRequest, ChatResponse, HealthResponse, MCPTestResponse
from .session_store import SessionStore


app = FastAPI(title="Muslim Bot", version="0.1.0")
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "muslim_bot" / "static"), name="static")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    return SessionStore(max_turns=get_settings().session_max_turns)


@lru_cache(maxsize=1)
def get_runner() -> MuslimBotRunner:
    return MuslimBotRunner(settings=get_settings())


@lru_cache(maxsize=1)
def get_config_store() -> RuntimeConfigStore:
    return RuntimeConfigStore(settings=get_settings())


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
    return HealthResponse(
        status=status,
        llm_configured=runtime_config.llm.is_configured,
        quran_db_ready=quran_ready,
        hadith_db_ready=hadith_ready,
        quran_api_dir=str(settings.quran_api_dir),
        hadith_api_dir=str(settings.hadith_api_dir),
        runtime_config_path=str(settings.runtime_config_path),
        enabled_mcp_servers=sum(1 for server in runtime_config.mcp_servers if server.enabled),
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
) -> ChatResponse:
    runtime_config = RuntimeConfigStore(settings).load()
    if not runtime_config.llm.is_configured:
        raise HTTPException(status_code=503, detail="LLM connection is not configured")

    session_id = store.ensure_session(request.session_id)
    history_prompt = store.history_prompt(session_id, request.message)
    try:
        reply = await runner.ask(
            request.message,
            session_id=session_id,
            history_prompt=history_prompt,
            language=request.language,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Agent request timed out") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    store.append(session_id, request.message, reply.answer)
    return ChatResponse(
        session_id=session_id,
        answer=reply.answer,
        references=reply.references,
        used_tools=reply.used_tools,
        warnings=reply.warnings,
    )
