# SkinTag API

FastAPI backend for skin lesion classification using MedSigLIP embeddings.

## Prerequisites

1. Train the classifier first (from project root):
   ```bash
   make train
   ```
   This creates `results/cache/classifier.pkl`.

2. Install API dependencies:
   ```bash
   pip install -r api/requirements.txt
   ```

## Running the API

### Development
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or run directly:
```bash
python api/main.py
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Classify Image
```bash
curl -X POST http://localhost:8000/classify \
  -F "file=@path/to/skin_image.jpg"
```

Response:
```json
{
  "prediction": 0,
  "confidence": 0.92,
  "class_name": "benign",
  "probabilities": {
    "benign": 0.92,
    "malignant": 0.08
  }
}
```

## iOS App Configuration

When testing with the iOS app on a physical device:

1. Find your Mac's IP address:
   ```bash
   ipconfig getifaddr en0
   ```

2. Start the API on all interfaces:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. In the iOS app Settings, set the API endpoint to:
   ```
   http://<your-mac-ip>:8000
   ```

## API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
