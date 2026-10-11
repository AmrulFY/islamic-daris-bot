# Islamic Daris Bot

AI agent for Islamic source lookup and educational answers in Indonesian. It uses configurable MCP servers for Quran and Hadith source access:

- `../quran-api-go/cmd/mcp`
- `../hadith-api-go-run/cmd/mcp`

The bot exposes both a CLI and a small FastAPI service.

## Setup

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
# set ADMIN_TOKEN before exposing the admin UI beyond localhost
make install
make bootstrap
```

If PyPI fails with a local certificate verification error, use:

```bash
make install PIP_FLAGS="--trusted-host pypi.org --trusted-host files.pythonhosted.org"
```

`make bootstrap` copies the valid Quran SQLite database into `.data/quran.db` and seeds Hadith data into `.data/hadith.db`. Hadith seeding downloads data from `fawazahmed0/hadith-api` through jsDelivr, so it needs network access the first time.

The local Go MCP projects declare newer Go toolchains than the system `go` binary. Keep `GOTOOLCHAIN=auto` enabled so `go run ./cmd/mcp` can resolve the required toolchain.

## Run

```bash
make run-cli
make chat q="ayat tentang sabar"
make run-api
```

Admin UI:

```text
http://127.0.0.1:8000
```

Use the `ADMIN_TOKEN` value from `.env` to edit the LLM connection and MCP servers. Settings are saved to `.data/runtime_config.json`; secret values are masked when read back by the UI.

API:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"hadis tentang niat"}'
```

Streaming (Server-Sent Events with `delta`, `tool`, `done`, and `error` events):

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H 'content-type: application/json' \
  -d '{"message":"hadis tentang niat"}'
```

## Behavior

Islamic Daris Bot is educational by default. It searches the Quran/Hadith MCP tools for religious claims, cites references in the answer, and avoids presenting itself as a mufti or issuing formal fatwas.

Direct lookups (a hadith number, a `surah:ayah` reference, "ayat kursi", or a
random verse request) are answered straight from the local databases without
touching the LLM. Complete agent answers to first-time questions are cached in
the local database (`CHAT_CACHE_ENABLED`, `ANSWER_CACHE_TTL_DAYS`).

## OpenClaw VPS

Docker/Compose deployment files live in `deploy/`:

```bash
make compose-config
make compose-up
```

The Compose file binds `127.0.0.1:8000` on the VPS. Use SSH tunnel, Tailscale, or a reverse proxy for access. See `deploy/OPENCLAW_VPS.md`.
