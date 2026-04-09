FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.12 python3.12-venv curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml .
COPY service.py .
COPY config.yaml .

# Copy external megaDNA package
COPY ../external/megaDNA /external/megaDNA

RUN uv sync --frozen

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
