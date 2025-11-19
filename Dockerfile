# Use a PyTorch image that already has torch + CUDA + cuDNN
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install Python deps, avoid caching wheels to keep layers smaller
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "transformers[torch]==4.45.0" \
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
