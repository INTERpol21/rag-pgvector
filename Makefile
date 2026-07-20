.PHONY: install install-dev run test lint typecheck eval lock

# uv is the source of truth; requirements*.txt are exported from uv.lock for Docker/pip users.
install:
	uv sync --frozen --no-dev

install-dev:
	uv sync --frozen

run:
	uv run uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8081

test:
	uv run pytest

lint:
	uv run ruff check app tests evals

typecheck:
	uv run mypy app

eval:
	uv run python evals/run_evals.py --min-hit-rate 0.7

# Regenerate uv.lock and the exported requirements files after editing pyproject.toml.
lock:
	uv lock
	uv export --frozen --no-hashes --no-dev --no-emit-project -o requirements.txt
	uv export --frozen --no-hashes --only-dev --no-emit-project -o requirements-dev.txt
