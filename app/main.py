"""FastAPI backend for SkinTag triage web application.

Provides upload endpoint for skin lesion images, runs SigLIP embedding +
classifier inference, and returns triage assessment results.
"""

import os
import sys
from pathlib import Path

# Add project root
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import io
import pickle
import yaml
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from src.model.embeddings import EmbeddingExtractor
from src.model.triage import TriageSystem
from src.utils.model_hub import download_model_from_hf, download_e2e_model_from_hf, get_model_config

app = FastAPI(title="SkinTag", description="AI-powered skin lesion triage screening tool")

# Global state (loaded on startup)
_state = {
    "extractor": None,
    "classifier": None,
    "condition_classifier": None,  # 10-class condition estimator
    "e2e_model": None,  # End-to-end fine-tuned model (if available)
    "triage": None,
    "config": None,
    "inference_mode": None,  # "e2e" or "embedding+head"
}


@app.on_event("startup")
async def load_models():
    """Load models and config on server startup.

    Downloads models from Hugging Face Hub if enabled, otherwise loads from local cache.
    Prefers fine-tuned end-to-end model if available (better accuracy),
    falls back to embedding extractor + classifier head.
    """
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        _state["config"] = yaml.safe_load(f)

    cache_dir = PROJECT_ROOT / "results" / "cache"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_hf = os.getenv("USE_HF_MODELS", "false").lower() in ("true", "1", "yes")

    # Download from Hugging Face if enabled
    if use_hf:
        print("Downloading models from Hugging Face Hub...")
        try:
            model_config = get_model_config()
            repo_id = model_config["repo_id"]

            # Try loading fine-tuned end-to-end model first (best accuracy)
            try:
                from src.model.deep_classifier import EndToEndClassifier
                model_dir = download_e2e_model_from_hf(
                    repo_id=repo_id,
                    cache_subdir="skintag"
                )
                _state["e2e_model"] = EndToEndClassifier.load_for_inference(str(model_dir), device=device)
                _state["inference_mode"] = "e2e"
                print(f"✓ Loaded fine-tuned model from HF: {repo_id} (device={device})")

                # Still load embedding extractor for condition classifier
                _state["extractor"] = EmbeddingExtractor(device=device)

            except Exception as e:
                print(f"E2E model not available, trying classifier files: {e}")

                # Fall back to embedding extractor + classifier head
                classifier_path = download_model_from_hf(
                    repo_id=repo_id,
                    filename=model_config["classifier_filename"],
                    cache_subdir="skintag"
                )
                with open(classifier_path, "rb") as f:
                    _state["classifier"] = pickle.load(f)
                print(f"✓ Loaded classifier from HF: {classifier_path.name}")

                _state["extractor"] = EmbeddingExtractor(device=device)
                _state["inference_mode"] = "embedding+head"

            # Download condition classifier (optional)
            try:
                cond_path = download_model_from_hf(
                    repo_id=repo_id,
                    filename=model_config["condition_classifier_filename"],
                    cache_subdir="skintag"
                )
                with open(cond_path, "rb") as f:
                    _state["condition_classifier"] = pickle.load(f)
                print(f"✓ Loaded condition classifier from HF: {cond_path.name}")
            except Exception as e:
                print(f"Condition classifier not available: {e}")

            print(f"✓ Models loaded from Hugging Face (mode={_state['inference_mode']}, device={device})")

        except Exception as e:
            print(f"Failed to load from Hugging Face: {e}")
            print("Falling back to local cache...")
            use_hf = False

    # Load from local cache
    if not use_hf:
        # Try loading fine-tuned end-to-end model first
        e2e_dir = cache_dir / "finetuned_model"
        if (e2e_dir / "config.json").exists():
            try:
                from src.model.deep_classifier import EndToEndClassifier
                _state["e2e_model"] = EndToEndClassifier.load_for_inference(str(e2e_dir), device=device)
                _state["inference_mode"] = "e2e"
                print(f"Loaded fine-tuned end-to-end model from {e2e_dir}")
            except Exception as e:
                print(f"Failed to load e2e model: {e}, falling back to embedding+head")

        # Fall back to embedding extractor + pickled classifier
        if _state["inference_mode"] is None:
            for model_name in ["classifier_deep_mlp.pkl", "classifier_logistic_regression.pkl",
                                "classifier_deep.pkl", "classifier_logistic.pkl", "classifier.pkl"]:
                model_path = cache_dir / model_name
                if model_path.exists():
                    with open(model_path, "rb") as f:
                        _state["classifier"] = pickle.load(f)
                    print(f"Loaded classifier: {model_name}")
                    break

            if _state["classifier"] is None:
                print("WARNING: No trained classifier found. Set USE_HF_MODELS=true or run train.py first.")

            _state["extractor"] = EmbeddingExtractor(device=device)
            _state["inference_mode"] = "embedding+head"
            print(f"Embedding extractor ready (device={device})")

        # Load condition classifier (10-class)
        cond_path = cache_dir / "classifier_condition.pkl"
        if cond_path.exists():
            with open(cond_path, "rb") as f:
                _state["condition_classifier"] = pickle.load(f)
            print(f"Loaded condition classifier: {cond_path.name}")
        else:
            print("No condition classifier found (condition estimation disabled)")

    # Load triage system
    triage_config = _state["config"].get("triage", {})
    _state["triage"] = TriageSystem(triage_config)
    print(f"Triage system ready (inference mode: {_state['inference_mode']})")


@app.on_event("shutdown")
async def cleanup():
    if _state["extractor"] is not None:
        _state["extractor"].unload_model()


@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze an uploaded skin lesion image.

    Returns triage assessment with risk score, urgency tier, recommendation.
    """
    if _state["inference_mode"] == "e2e" and _state["e2e_model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if _state["inference_mode"] == "embedding+head" and _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Classify — use end-to-end model or embedding+head
    if _state["inference_mode"] == "e2e":
        proba = _state["e2e_model"].predict_proba([image])
    else:
        extractor = _state["extractor"]
        embedding = extractor.extract([image])  # (1, 1152)
        clf = _state["classifier"]
        proba = clf.predict_proba(embedding.numpy())

    mal_prob = float(proba[0, 1]) if proba.ndim == 2 else float(proba[0])

    # Triage assessment
    triage = _state["triage"]
    result = triage.assess(mal_prob)

    response = {
        "risk_score": round(result.risk_score, 4),
        "urgency_tier": result.urgency_tier,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "disclaimer": result.disclaimer,
        "probabilities": {
            "benign": round(1 - mal_prob, 4),
            "malignant": round(mal_prob, 4),
        },
    }

    # Condition estimation (10-class)
    cond_clf = _state.get("condition_classifier")
    if cond_clf is not None:
        try:
            from src.data.taxonomy import CONDITION_NAMES, Condition

            if _state["inference_mode"] == "e2e":
                # For e2e, re-extract embedding for condition head
                emb = _state["extractor"].extract([image]) if _state["extractor"] else None
                cond_input = emb.numpy() if emb is not None else None
            else:
                cond_input = embedding.numpy()

            if cond_input is not None:
                cond_proba = cond_clf.predict_proba(cond_input)
                if cond_proba.ndim == 2:
                    top_indices = np.argsort(cond_proba[0])[::-1][:3]
                    top_condition = int(top_indices[0])
                    condition_probs = []
                    for idx in top_indices:
                        cond_enum = Condition(int(idx))
                        condition_probs.append({
                            "condition": CONDITION_NAMES.get(cond_enum, f"Class {idx}"),
                            "probability": round(float(cond_proba[0, idx]), 4),
                        })

                    cond_enum = Condition(top_condition)
                    response["condition_estimate"] = CONDITION_NAMES.get(cond_enum, f"Class {top_condition}")
                    response["condition_probabilities"] = condition_probs
        except Exception:
            pass  # Condition estimation is optional; don't fail the request

    return JSONResponse(response)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "inference_mode": _state["inference_mode"],
        "model_loaded": (_state["e2e_model"] is not None) or (_state["classifier"] is not None),
        "device": (
            _state["extractor"].device if _state["extractor"]
            else (_state["e2e_model"].device if _state["e2e_model"] else "unknown")
        ),
    }


# Serve static files and frontend (only if directory exists)
_static_dir = APP_DIR / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = APP_DIR / "templates" / "index.html"
    if index_path.exists():
        return index_path.read_text()
    # Fallback: serve from static
    static_index = APP_DIR / "static" / "index.html"
    if static_index.exists():
        return static_index.read_text()
    return "<h1>SkinTag</h1><p>Frontend not found. Place index.html in app/templates/</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
