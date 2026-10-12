# Deploy Islamic Daris Bot on an OpenClaw VPS

This app is designed to run beside an OpenClaw gateway on the same Linux VPS. It binds to loopback by default, keeps runtime state in a Docker volume, and exposes the admin UI through SSH tunnel, Tailscale, or a reverse proxy you control.

## VPS Setup

```bash
sudo apt update
sudo apt install -y git curl ca-certificates
docker --version
docker compose version
```

Copy the `ydgi` workspace to the server so these sibling folders are present:

```text
ydgi/
  muslim_bot/
  quran-api-go/
  hadith-api-go-run/
```

Create the app env file:

```bash
cd /opt/ydgi/muslim_bot
cp .env.example .env
chmod 600 .env
```

Set at minimum:

```bash
ADMIN_TOKEN=replace-with-a-long-random-token
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.6
```

Recommended extras (all optional, with defaults):

```bash
AGENT_MAX_TURNS=10
SESSION_RETENTION_DAYS=7
CHAT_RATE_LIMIT_PER_MINUTE=30
MCP_STARTUP_TIMEOUT_SECONDS=45
TELEGRAM_BOT_TOKEN=123456:ABC-...   # from @BotFather; enables the telegram-bot service
```

`SESSION_DB_PATH` is already set to `/app/data/sessions.db` by the Compose file
so chat history survives container restarts. If your LLM gateway only supports
Chat Completions (not the Responses API), also set `OPENAI_USE_RESPONSES=false`.

## Start

Run Compose from the `muslim_bot` directory:

```bash
make compose-config
make compose-up
make compose-logs
```

The first start seeds the Hadith SQLite database from jsDelivr. That can take a few minutes and requires outbound HTTPS.

After seeding, the start-up script also repairs the seeded Hadith database
(the Go seeder leaves the `chapters` table and the FTS index empty, which makes
the `search_hadith` MCP tool return nothing). It downloads the six edition files
from jsDelivr once; if that fails, start-up continues with a warning and the
repair is retried on the next start.

### Semantic search index (first start)

Answers use keyword search plus an embedding-based semantic search. The vector
index (`/app/data/semantic.db`, about 80 MB, on the data volume) is built
automatically in the background on the first start:

- It embeds about 40 thousand texts through your LLM gateway's embeddings
  endpoint (`EMBEDDING_MODEL`, default `text-embedding-3-small`). That takes
  roughly 7-10 minutes and costs a few cents.
- While it builds, the bot answers normally using keyword search only. If the
  process restarts mid-build, the build resumes where it stopped.
- Check progress with `GET /health`: `semantic_index` is `building`, then
  `ready`. It is informational only and never makes the health check fail.
- The gateway must serve the embeddings endpoint. If it does not, set
  `SEMANTIC_SEARCH_ENABLED=false`.
- To skip the build, copy an existing `semantic.db` into the volume (it must be
  built with the same `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS`), or set
  `SEMANTIC_AUTO_INDEX=false` to never build automatically
  (`docker compose exec muslim-bot python -m muslim_bot.semantic_index` builds it by hand).
- Rebuild the index (delete `semantic.db`) if the Quran or Hadith database is replaced.

If `TELEGRAM_BOT_TOKEN` is set, a second container (`muslim-bot-telegram`)
starts the Telegram polling bot automatically. Without a token it exits; the
web API is unaffected.

## Access

The Compose file binds the app to `127.0.0.1:8021` on the VPS.

SSH tunnel:

```bash
ssh -L 8021:127.0.0.1:8021 user@your-vps
```

Then open:

```text
http://127.0.0.1:8021
```

For public HTTPS, place Nginx, Caddy, or Tailscale Serve in front of the loopback app. Keep `ADMIN_TOKEN` required for config changes.

## Operations

```bash
docker compose -f deploy/docker-compose.openclaw.yml ps
docker compose -f deploy/docker-compose.openclaw.yml restart muslim-bot
docker compose -f deploy/docker-compose.openclaw.yml pull
docker compose -f deploy/docker-compose.openclaw.yml up -d --build
```

Back up the Docker volume because it contains the runtime config, seeded databases and the semantic index:

```bash
docker run --rm -v muslim_bot_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/muslim_bot_data.tgz -C /data .
```
