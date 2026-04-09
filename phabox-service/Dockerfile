FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY environment.yaml .
RUN conda env create -f environment.yaml && conda clean -afy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
COPY service.py .
COPY config.yaml .

RUN uv sync --frozen

EXPOSE 8005
CMD ["uv", "run", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8005"]
