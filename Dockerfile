FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install Elan/Lean
RUN apt update -y \
    && apt install -y curl git gcc vim

# Install Python stuff
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY README.md pyproject.toml requirements.lock requirements-dev.lock .
RUN pip install uv && uv pip install typer torch==2.5.1 datasets pandas scikit-learn transformers openai trl peft vllm unsloth==2025.2.14 unsloth_zoo==2025.2.7 wandb msgspec
COPY src src/

CMD ["python", "src/formalize/align.py", "--help"]
