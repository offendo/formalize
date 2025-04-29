FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

# Install Elan/Lean
RUN apt update -y \
    && apt install -y curl git gcc vim

# Install Python stuff
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
# COPY README.md pyproject.toml requirements.lock requirements-dev.lock .
RUN pip install uv && \
    uv pip install --break-system-packages --system typer more_itertools icecream torchao datasets pandas transformers peft wandb msgspec
RUN uv pip install --break-system-packages --system --no-build-isolation axolotl[deepspeed]

COPY src src/
