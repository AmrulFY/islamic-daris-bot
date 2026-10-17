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
http://127.0.0.1:8021
```

Use the `ADMIN_TOKEN` value from `.env` to edit the LLM connection and MCP servers. Settings are saved to `.data/runtime_config.json`; secret values are masked when read back by the UI.

API:

```bash
curl -X POST http://localhost:8021/chat \
  -H 'content-type: application/json' \
  -d '{"message":"hadis tentang niat"}'
```

Streaming (Server-Sent Events with `delta`, `tool`, `done`, and `error` events):

```bash
curl -N -X POST http://localhost:8021/chat/stream \
  -H 'content-type: application/json' \
  -d '{"message":"hadis tentang niat"}'
```

## Behavior

Islamic Daris Bot is educational by default. It searches the Quran/Hadith MCP tools for religious claims, cites references in the answer, and avoids presenting itself as a mufti or issuing formal fatwas.

Direct lookups (a hadith number, a `surah:ayah` reference, "ayat kursi", or a
random verse request) are answered straight from the local databases without
touching the LLM. Complete agent answers to first-time questions are cached in
the local database (`CHAT_CACHE_ENABLED`, `ANSWER_CACHE_TTL_DAYS`).

## Calendar and prayer times

- **Hijri date and holidays** come from the unofficial MABIMS API (`api.mabims.dev`, based on the
  Kemenag calendar tables). When it is down the bot falls back to a local Umm al-Qura calculation and
  says so, because that can differ by a day from Kemenag. `/hijriyah` (Telegram) and questions like
  "tanggal hijriyah hari ini" are answered without the LLM; questions such as "kapan puasa tahun ini?"
  give the LLM a calendar block so it never has to guess a date. Settings: `HIJRI_API_*`.
- **Prayer times** come from Aladhan (method 20, Kementerian Agama RI) and are computed locally when it
  is down (agrees with Aladhan to within a minute). Each Telegram user saves a place with `/lokasi`
  (share a location pin, or type a city or `Kota, Provinsi`) and then uses `/jadwal` or just asks
  ("maghrib jam berapa?"). Only coordinates rounded to about 1 km are stored, with the session's other
  settings: `/lokasi hapus` or `/lupa` removes them, and they expire with an inactive session
  (unless reminders are on, see below). Settings: `PRAYER_*`.
- **Reminders** (`/pengingat on|off`, Telegram only) are sent exactly at Imsak, Subuh, Terbit, Dzuhur,
  Ashar, Maghrib and Isya of the saved location; `/pengingat imsak off` mutes a single time. The bot
  checks every minute, sends a missed one for up to 5 minutes, and never repeats one (the server
  remembers the last reminder sent). Reminders need a saved location, and removing the location turns
  them off. A user with reminders on is not purged for inactivity (their chat history still is), and a
  chat that blocked the bot has its reminders turned off. **The bot needs `ADMIN_TOKEN`** (the same value
  as the API's) to read who has reminders on: that list is admin-only because it is a list of locations.
  Without it `/pengingat on` says reminders are unavailable instead of promising what cannot be sent.
- The city list (`muslim_bot/data/id_cities.csv`, 483 Indonesian places) is built by
  `scripts/build_id_cities.py` from [GeoNames](https://www.geonames.org), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## OpenClaw VPS

Docker/Compose deployment files live in `deploy/`:

```bash
make compose-config
make compose-up
```

The Compose file binds `127.0.0.1:8021` on the VPS. Use SSH tunnel, Tailscale, or a reverse proxy for access. See `deploy/OPENCLAW_VPS.md`.
