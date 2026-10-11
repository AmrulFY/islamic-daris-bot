from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default.resolve()
    return Path(raw).expanduser().resolve()


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_or_none_from_env(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _bool_from_env(name: str, default: bool) -> bool:
    value = _bool_or_none_from_env(name)
    return default if value is None else value


def _args_from_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return []
    return shlex.split(raw)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    openai_base_url: str | None
    openai_organization: str | None
    openai_project: str | None
    openai_use_responses: bool | None
    admin_token: str | None
    runtime_config_path: Path
    quran_source_db_path: Path
    quran_api_dir: Path
    quran_db_path: Path
    quran_mcp_command: str
    quran_mcp_args: list[str]
    quran_mcp_cwd: Path
    hadith_api_dir: Path
    hadith_db_path: Path
    hadith_mcp_command: str
    hadith_mcp_args: list[str]
    hadith_mcp_cwd: Path
    hadith_seeder_command: str
    hadith_seeder_args: list[str]
    hadith_seeder_cwd: Path
    mcp_startup_timeout_seconds: int
    agent_timeout_seconds: int
    agent_max_turns: int
    session_max_turns: int
    session_db_path: Path
    session_retention_days: float
    chat_rate_limit_per_minute: int
    telegram_bot_token: str | None
    telegram_api_base: str
    telegram_proxy: str | None
    telegram_connect_timeout: float
    telegram_read_timeout: float
    chat_cache_enabled: bool
    answer_cache_ttl_days: float
    mcp_health_check_interval_seconds: float = 30.0
    agent_parallel_tool_calls: bool = True
    agent_pipeline_enabled: bool = True
    semantic_search_enabled: bool = True
    semantic_auto_index: bool = True
    semantic_db_path: Path | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 512

    @property
    def has_openai_api_key(self) -> bool:
        return bool(self.openai_api_key)


def load_settings(env_file: Path | None = PROJECT_ROOT / ".env") -> Settings:
    if env_file and env_file.exists():
        load_dotenv(env_file, override=False)

    data_dir = PROJECT_ROOT / ".data"
    quran_api_dir = _path_from_env("QURAN_API_DIR", WORKSPACE_ROOT / "quran-api-go")
    hadith_api_dir = _path_from_env("HADITH_API_DIR", WORKSPACE_ROOT / "hadith-api-go-run")
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.6"),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        openai_organization=os.getenv("OPENAI_ORGANIZATION") or None,
        openai_project=os.getenv("OPENAI_PROJECT") or None,
        openai_use_responses=_bool_or_none_from_env("OPENAI_USE_RESPONSES"),
        admin_token=os.getenv("ADMIN_TOKEN") or None,
        runtime_config_path=_path_from_env("RUNTIME_CONFIG_PATH", data_dir / "runtime_config.json"),
        quran_source_db_path=_path_from_env("QURAN_SOURCE_DB_PATH", WORKSPACE_ROOT / "quran-api-go" / "data" / "quran.db"),
        quran_api_dir=quran_api_dir,
        quran_db_path=_path_from_env("QURAN_DB_PATH", data_dir / "quran.db"),
        quran_mcp_command=os.getenv("QURAN_MCP_COMMAND", "go"),
        quran_mcp_args=_args_from_env("QURAN_MCP_ARGS", ["run", "./cmd/mcp"]),
        quran_mcp_cwd=_path_from_env("QURAN_MCP_CWD", quran_api_dir),
        hadith_api_dir=hadith_api_dir,
        hadith_db_path=_path_from_env("HADITH_DB_PATH", data_dir / "hadith.db"),
        hadith_mcp_command=os.getenv("HADITH_MCP_COMMAND", "go"),
        hadith_mcp_args=_args_from_env("HADITH_MCP_ARGS", ["run", "./cmd/mcp"]),
        hadith_mcp_cwd=_path_from_env("HADITH_MCP_CWD", hadith_api_dir),
        hadith_seeder_command=os.getenv("HADITH_SEEDER_COMMAND", "go"),
        hadith_seeder_args=_args_from_env("HADITH_SEEDER_ARGS", ["run", "./cmd/seeder"]),
        hadith_seeder_cwd=_path_from_env("HADITH_SEEDER_CWD", hadith_api_dir),
        mcp_startup_timeout_seconds=_int_from_env("MCP_STARTUP_TIMEOUT_SECONDS", 45),
        agent_timeout_seconds=_int_from_env("AGENT_TIMEOUT_SECONDS", 300),
        agent_max_turns=_int_from_env("AGENT_MAX_TURNS", 10),
        session_max_turns=_int_from_env("SESSION_MAX_TURNS", 12),
        session_db_path=_path_from_env("SESSION_DB_PATH", data_dir / "sessions.db"),
        session_retention_days=_float_from_env("SESSION_RETENTION_DAYS", 7),
        chat_rate_limit_per_minute=_int_from_env("CHAT_RATE_LIMIT_PER_MINUTE", 30),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_api_base=os.getenv("TELEGRAM_API_BASE") or "http://127.0.0.1:8000",
        telegram_proxy=os.getenv("TELEGRAM_PROXY") or None,
        telegram_connect_timeout=_float_from_env("TELEGRAM_CONNECT_TIMEOUT", 10.0),
        telegram_read_timeout=_float_from_env("TELEGRAM_READ_TIMEOUT", 30.0),
        chat_cache_enabled=_bool_from_env("CHAT_CACHE_ENABLED", True),
        answer_cache_ttl_days=_float_from_env("ANSWER_CACHE_TTL_DAYS", 7),
        mcp_health_check_interval_seconds=_float_from_env("MCP_HEALTH_CHECK_INTERVAL_SECONDS", 30.0),
        agent_parallel_tool_calls=_bool_from_env("AGENT_PARALLEL_TOOL_CALLS", True),
        agent_pipeline_enabled=_bool_from_env("AGENT_PIPELINE_ENABLED", True),
        semantic_search_enabled=_bool_from_env("SEMANTIC_SEARCH_ENABLED", True),
        semantic_auto_index=_bool_from_env("SEMANTIC_AUTO_INDEX", True),
        semantic_db_path=_path_from_env("SEMANTIC_DB_PATH", data_dir / "semantic.db"),
        embedding_model=os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",
        embedding_dimensions=_int_from_env("EMBEDDING_DIMENSIONS", 512),
    )
