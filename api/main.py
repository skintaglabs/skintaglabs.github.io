"""FastAPI backend for SkinTag skin lesion classification."""

import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.embeddings import EmbeddingExtractor, get_device

# Global model references
extractor: EmbeddingExtractor | None = None
classifier = None
demo_mode = False
CLASS_NAMES = ["benign", "malignant"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup at shutdown."""
    global extractor, classifier, demo_mode

    # Paths
    classifier_path = Path(__file__).parent.parent / "results" / "cache" / "classifier.pkl"

    # Load classifier if available
    if classifier_path.exists():
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
        print(f"Loaded classifier from {classifier_path}")
    else:
        demo_mode = True
        print(f"WARNING: No classifier found at {classifier_path}")
        print("Running in DEMO MODE - predictions are based on embeddings only")

    # Load embedding extractor
    device = get_device()
    extractor = EmbeddingExtractor(device=device)
    extractor.load_model()
    print(f"Loaded EmbeddingExtractor on {device}")

    yield

    # Cleanup
    if extractor:
        extractor.unload_model()
    print("Models unloaded")


app = FastAPI(
    title="SkinTag API",
    description="Skin lesion classification using MedSigLIP embeddings",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": extractor is not None,
        "classifier_loaded": classifier is not None,
        "demo_mode": demo_mode,
        "device": extractor.device if extractor else "unknown",
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """
    Classify a skin lesion image.

    Accepts multipart/form-data with an image file.
    Returns prediction, confidence, and class name.
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image/*"
        )

    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        # Extract embeddings
        with torch.no_grad():
            embeddings = extractor.extract([image])

        if classifier is not None:
            # Production mode: use trained classifier
            prediction = int(classifier.predict(embeddings)[0])
            probabilities = classifier.predict_proba(embeddings)[0]
            confidence = float(probabilities[prediction])
        else:
            # Demo mode: use embedding norm as a proxy score
            # This is NOT a real prediction - just for API testing
            emb_np = embeddings.numpy()
            # Use a simple hash of embedding to generate consistent pseudo-predictions
            score = float(np.mean(emb_np) + 0.5)  # Normalize around 0.5
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            prediction = 1 if score > 0.5 else 0
            probabilities = [1 - score, score]
            confidence = float(probabilities[prediction])

        class_name = CLASS_NAMES[prediction]

        response = {
            "prediction": prediction,
            "confidence": confidence,
            "class_name": class_name,
            "probabilities": {
                CLASS_NAMES[i]: float(p) for i, p in enumerate(probabilities)
            },
        }

        if demo_mode:
            response["warning"] = "DEMO MODE: Predictions are not from a trained model"

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
