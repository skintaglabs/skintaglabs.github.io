FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (CPU PyTorch — for GPU use Dockerfile.gpu)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY run_pipeline.py .

# Pre-download SigLIP model into the image so first inference is fast
# (remove this line if you prefer to mount HF cache as a volume instead)
RUN python -c "\
from transformers import AutoModel, AutoImageProcessor; \
m = 'google/siglip-so400m-patch14-384'; \
AutoImageProcessor.from_pretrained(m); \
AutoModel.from_pretrained(m); \
print('SigLIP model cached')"

EXPOSE 8000

# Default: launch the web app
# Mount trained model: -v /path/to/results:/app/results
# For full pipeline: override CMD with "python run_pipeline.py --no-app"
CMD ["python", "run_pipeline.py", "--app-only"]
