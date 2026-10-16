from __future__ import annotations

from pydantic import BaseModel, Field

from .runtime_config import RuntimeConfig


class Reference(BaseModel):
    type: str = Field(pattern="^(quran|hadith)$")
    label: str
    source_id: str | None = None
    verified: bool | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    language: str = Field(default="id", pattern="^(id|en)$")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    references: list[Reference] = Field(default_factory=list)
    used_tools: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PrefsUpdate(BaseModel):
    """Only the fields present in the request are changed; ``null`` resets a field."""

    language: str | None = Field(default=None, pattern="^(id|en)$")
    style: str | None = Field(default=None, pattern="^(ringkas|lengkap)$")
    note: str | None = Field(default=None, max_length=200)
    history_turns: int | None = Field(default=None, ge=0)


class PrefsResponse(BaseModel):
    session_id: str
    language: str | None = None
    style: str | None = None
    note: str | None = None
    history_turns: int | None = None
    max_history_turns: int


class HistoryClearResponse(BaseModel):
    session_id: str
    deleted_turns: int


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    quran_db_ready: bool
    hadith_db_ready: bool
    quran_api_dir: str
    hadith_api_dir: str
    runtime_config_path: str
    enabled_mcp_servers: int
    semantic_index: str = "disabled"  # ready | building | missing | disabled


class AdminConfigResponse(BaseModel):
    config: RuntimeConfig
    saved: bool


class MCPToolSummary(BaseModel):
    id: str
    name: str
    enabled: bool
    tools: list[str] = Field(default_factory=list)
    error: str | None = None


class MCPTestResponse(BaseModel):
    servers: list[MCPToolSummary]
