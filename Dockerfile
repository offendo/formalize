FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Install Elan/Lean
RUN apt update -y \
    && apt install -y curl git

# Install Python stuff
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1
COPY README.md pyproject.toml requirements.lock requirements-dev.lock .
RUN pip install datasets pandas scikit-learn transformers openai trl peft vllm unsloth==2025.2.14 unsloth_zoo==2025.2.7
COPY src src/

CMD ["python", "src/formalize/align.py", "--help"]
