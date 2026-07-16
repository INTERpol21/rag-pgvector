.PHONY: install install-dev run test lint eval

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8081

test:
	python -m pytest

lint:
	ruff check app tests evals

eval:
	python evals/run_evals.py --min-hit-rate 0.7
