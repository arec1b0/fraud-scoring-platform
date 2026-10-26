FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --locked --no-dev --no-editable \
    && .venv/bin/python -c \
        "from fraud_scoring.api.app import app; assert app is not None"

ENV PATH="/app/.venv/bin:$PATH"

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "fraud_scoring.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
