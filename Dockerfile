FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install Elan/Lean
RUN apt update -y \
    && apt install -y curl git gcc vim

# Install Python stuff
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY README.md pyproject.toml requirements.lock requirements-dev.lock .
RUN pip install uv && \
  conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit && \
  uv pip install --system -U typer more_itertools icecream torchao torch==2.5.1 datasets pandas scikit-learn \
                             transformers openai trl==0.15.2 peft vllm \
                             unsloth==2025.2.14 unsloth_zoo==2025.2.7 wandb msgspec "axolotl[flash-attn,deepspeed]"
COPY src src/

CMD ["python", "src/formalize/align.py", "--help"]
