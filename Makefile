PYTHON ?= 3.13

.PHONY: setup dev test

setup:
	uv python install $(PYTHON)
	uv sync --group dev

dev:
	uv run uvicorn sales_agent.main:app --reload

test:
	uv run pytest -q
