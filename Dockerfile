# FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM nvcr.io/nvidia/pytorch:25.03-py3

# Install Elan/Lean
RUN apt update -y \
    && apt install -y curl git gcc vim

# Install Python stuff
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
# COPY README.md pyproject.toml requirements.lock requirements-dev.lock .
RUN pip install uv && \
    uv pip install --break-system-packages --system typer more_itertools icecream torchao datasets pandas transformers peft wandb msgspec
RUN uv pip install --break-system-packages --system --no-build-isolation axolotl flash-attn deepspeed

COPY src src/
