FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Install Elan/Lean
RUN apt update -y \
    && apt install -y curl git gcc vim

# Install Python stuff
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY install_cuda.sh .
RUN bash install_cuda.sh && \
    pip install uv && \
    uv pip install --break-system-packages --system typer more_itertools icecream torchao datasets pandas transformers peft wandb msgspec vllm && \
    uv pip install --break-system-packages --system --no-build-isolation axolotl deepspeed

COPY src src/
