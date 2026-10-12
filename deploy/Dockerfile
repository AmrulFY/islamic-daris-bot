# syntax=docker/dockerfile:1

FROM golang:1.25-bookworm AS go-builder

WORKDIR /src

COPY quran-api-go ./quran-api-go
COPY hadith-api-go-run ./hadith-api-go-run

RUN cd quran-api-go && go build -o /out/quran-mcp ./cmd/mcp
RUN cd hadith-api-go-run && go build -o /out/hadith-mcp ./cmd/mcp
RUN cd hadith-api-go-run && go build -o /out/hadith-seeder ./cmd/seeder


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8021 \
    RUNTIME_CONFIG_PATH=/app/data/runtime_config.json \
    QURAN_SOURCE_DB_PATH=/app/seed/quran.db \
    QURAN_DB_PATH=/app/data/quran.db \
    HADITH_DB_PATH=/app/data/hadith.db \
    SEMANTIC_DB_PATH=/app/data/semantic.db \
    QURAN_MCP_COMMAND=/app/bin/quran-mcp \
    QURAN_MCP_ARGS="" \
    QURAN_MCP_CWD=/app \
    HADITH_MCP_COMMAND=/app/bin/hadith-mcp \
    HADITH_MCP_ARGS="" \
    HADITH_MCP_CWD=/app/hadith-api-go-run \
    HADITH_SEEDER_COMMAND=/app/bin/hadith-seeder \
    HADITH_SEEDER_ARGS="" \
    HADITH_SEEDER_CWD=/app/hadith-api-go-run

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY muslim_bot/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY muslim_bot/muslim_bot ./muslim_bot
COPY muslim_bot/deploy/entrypoint.sh ./deploy/entrypoint.sh
COPY quran-api-go/data/quran.db ./seed/quran.db
COPY hadith-api-go-run/database ./hadith-api-go-run/database
COPY --from=go-builder /out/quran-mcp ./bin/quran-mcp
COPY --from=go-builder /out/hadith-mcp ./bin/hadith-mcp
COPY --from=go-builder /out/hadith-seeder ./bin/hadith-seeder

RUN chmod +x ./deploy/entrypoint.sh ./bin/quran-mcp ./bin/hadith-mcp ./bin/hadith-seeder \
    && mkdir -p /app/data

VOLUME ["/app/data"]
EXPOSE 8021

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${APP_PORT}/health" >/dev/null || exit 1

CMD ["./deploy/entrypoint.sh"]
