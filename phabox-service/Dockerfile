FROM continuumio/miniconda3:latest

WORKDIR /app
COPY environment.yaml .
RUN conda env create -f environment.yaml && conda clean -afy

# Make phabox2 and its dependencies (DIAMOND, BLAST, etc.) available on PATH
ENV PATH=/opt/conda/envs/phabox-tools/bin:$PATH

# Upgrade PyTorch to 2.11+ (2.4.1 from conda has executable-stack issue on Linux 6.8+)
# Downgrade pandas to 2.x (phabox2 has type-coercion issues with pandas 3.0+)
RUN pip install --no-cache-dir 'torch>=2.5' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir 'pandas<3'

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml .
COPY service.py .
COPY config.yaml .

RUN uv sync

EXPOSE 8005
CMD ["uv", "run", "uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8005"]
