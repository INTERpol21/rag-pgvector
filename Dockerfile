FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv/rag-pgvector

# requirements.txt is generated from pyproject.toml + uv.lock (see `make lock`).
COPY requirements.txt .
# BuildKit cache mount: a dependency bump re-downloads only the changed
# wheels instead of the whole set (--no-cache-dir kept nothing between builds).
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY app ./app
COPY data ./data
COPY evals ./evals
COPY migrations ./migrations

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /srv/rag-pgvector
USER appuser

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8081/healthz').status == 200 else 1)"

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8081"]
