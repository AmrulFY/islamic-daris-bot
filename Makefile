.PHONY: venv install bootstrap test run-api run-cli chat compose-config compose-up compose-logs clean

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PIP_FLAGS ?=

venv:
	python3 -m venv .venv

install: venv
	$(PIP) install $(PIP_FLAGS) --upgrade pip
	$(PIP) install $(PIP_FLAGS) -r requirements.txt

bootstrap:
	$(PYTHON) -m muslim_bot.bootstrap

test:
	$(PYTHON) -m pytest

run-api:
	$(PYTHON) -m uvicorn muslim_bot.api:app --host 0.0.0.0 --port 8021

run-cli:
	$(PYTHON) -m muslim_bot.cli

chat:
	$(PYTHON) -m muslim_bot.cli --once "$(q)"

compose-config:
	docker compose -f deploy/docker-compose.openclaw.yml config

compose-up:
	docker compose -f deploy/docker-compose.openclaw.yml up -d --build

compose-logs:
	docker compose -f deploy/docker-compose.openclaw.yml logs -f muslim-bot

clean:
	rm -rf .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
