FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN apt-get update && \
    apt-get install -y git python3-pip python3-tomli && \
    rm -rf /var/lib/apt/lists/*

RUN pip install evo2 fastapi uvicorn pyyaml loguru numpy

WORKDIR /workspace
COPY service.py config.yaml ./

EXPOSE 8001
CMD ["python", "service.py"]
