#!/bin/sh
set -eu

python -m muslim_bot.bootstrap

exec python -m uvicorn muslim_bot.api:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8021}"
