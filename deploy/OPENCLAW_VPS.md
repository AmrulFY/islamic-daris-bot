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

If `TELEGRAM_BOT_TOKEN` is set, a second container (`muslim-bot-telegram`)
starts the Telegram polling bot automatically. Without a token it exits; the
web API is unaffected.

## Access

The Compose file binds the app to `127.0.0.1:8000` on the VPS.

SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 user@your-vps
```

Then open:

```text
http://127.0.0.1:8000
```

For public HTTPS, place Nginx, Caddy, or Tailscale Serve in front of the loopback app. Keep `ADMIN_TOKEN` required for config changes.

## Operations

```bash
docker compose -f deploy/docker-compose.openclaw.yml ps
docker compose -f deploy/docker-compose.openclaw.yml restart muslim-bot
docker compose -f deploy/docker-compose.openclaw.yml pull
docker compose -f deploy/docker-compose.openclaw.yml up -d --build
```

Back up the Docker volume because it contains the runtime config and seeded databases:

```bash
docker run --rm -v muslim_bot_data:/data -v "$PWD":/backup alpine \
  tar czf /backup/muslim_bot_data.tgz -C /data .
```
