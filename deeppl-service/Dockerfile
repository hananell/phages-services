FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies.
# torch is already present in the base image; listing it here pins the extras.
RUN pip install \
    transformers>=4.30 \
    sentencepiece \
    fastapi>=0.100 \
    uvicorn>=0.22 \
    pyyaml>=6.0 \
    loguru>=0.7 \
    numpy>=1.24 \
    "pydantic>=2.0"

WORKDIR /workspace
COPY service.py config.yaml ./

# Model weights must be mounted at /model at runtime, e.g.:
#   docker run --gpus all \
#     -v /path/to/deeppl-weights:/model \
#     -p 8004:8004 \
#     deeppl-service
#
# The weights directory must contain:
#   config.json, pytorch_model.bin (or model.safetensors), vocab.txt
EXPOSE 8004

CMD ["python", "service.py"]
