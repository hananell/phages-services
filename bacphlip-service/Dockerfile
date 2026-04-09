FROM python:3.9-slim

# HMMER3 is required by bacphlip
RUN apt-get update && apt-get install -y hmmer git && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml .
COPY service.py .
COPY config_model.py .
COPY config.yaml .

RUN uv sync --frozen

EXPOSE 8003
CMD ["uv", "run", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8003"]
