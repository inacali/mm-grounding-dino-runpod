FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set python alias
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

# Create venv (optional but clean)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps
RUN pip install --upgrade pip

# Torch + Transformers + FastAPI stack
RUN pip install \
    "torch==2.3.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    "transformers[torch]==4.43.3" \
    accelerate \
    fastapi \
    "uvicorn[standard]" \
    pillow

# Copy app code
COPY app.py /app/app.py

# Expose port
EXPOSE 8000

# Default command (Runpod will override or use this)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

