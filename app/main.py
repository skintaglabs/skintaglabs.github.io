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
from fastapi.middleware.cors import CORSMiddleware

from src.model.embeddings import EmbeddingExtractor
from src.model.triage import TriageSystem
from src.utils.model_hub import download_model_from_hf, download_e2e_model_from_hf, get_model_config

app = FastAPI(title="SkinTag", description="AI-powered skin lesion triage screening tool")

# Enable CORS for GitHub Pages and local development
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.github\.io|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            revision = model_config.get("revision")

            # Try loading fine-tuned end-to-end model first (best accuracy)
            try:
                from src.model.deep_classifier import EndToEndClassifier
                model_dir = download_e2e_model_from_hf(
                    repo_id=repo_id,
                    revision=revision,
                    cache_subdir="skintag"
                )
                _state["e2e_model"] = EndToEndClassifier.load_for_inference(str(model_dir), device=device)
                _state["inference_mode"] = "e2e"
                print(f"✓ Loaded fine-tuned model from HF: {repo_id} (device={device})")

            except Exception as e:
                print(f"E2E model not available, trying classifier files: {e}")

                # Fall back to embedding extractor + classifier head
                classifier_path = download_model_from_hf(
                    repo_id=repo_id,
                    filename=model_config["classifier_filename"],
                    revision=revision,
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
                    revision=revision,
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
        # Check v2 path (siglip_finetuned subdir), then v1 path
        e2e_dir = cache_dir / "finetuned_model"
        v2_dir = e2e_dir / "siglip_finetuned"
        if (v2_dir / "config.json").exists():
            e2e_dir = v2_dir
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

        # Load condition classifier (10-class) — check v2 paths first
        cond_candidates = [
            cache_dir / "finetuned_model" / "classifiers" / "xgboost_condition.pkl",
            cache_dir / "classifier_condition.pkl",
        ]
        for cond_path in cond_candidates:
            if cond_path.exists():
                with open(cond_path, "rb") as f:
                    _state["condition_classifier"] = pickle.load(f)
                print(f"Loaded condition classifier: {cond_path}")
                break
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
    embedding = None
    if _state["inference_mode"] == "e2e":
        proba = _state["e2e_model"].predict_proba([image])
    else:
        embedding = _state["extractor"].extract([image])  # (1, 1152)
        proba = _state["classifier"].predict_proba(embedding.numpy())

    mal_prob = float(proba[0, 1]) if proba.ndim == 2 else float(proba[0])

    response = {
        "probabilities": {
            "benign": round(1 - mal_prob, 4),
            "malignant": round(mal_prob, 4),
        },
    }

    # Condition estimation (10-class) - adds triage_categories to response
    _add_condition_estimate(response, image, embedding)

    # Determine dominant triage category for context-aware recommendations
    dominant_category = None
    if "triage_categories" in response:
        cats = response["triage_categories"]
        dominant_category = max(cats, key=lambda k: cats[k]["probability"])

    # Triage assessment with category context
    result = _state["triage"].assess(mal_prob, dominant_category=dominant_category)

    response.update({
        "risk_score": round(result.risk_score, 4),
        "urgency_tier": result.urgency_tier,
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "disclaimer": result.disclaimer,
    })

    return JSONResponse(response)


def _add_condition_estimate(response: dict, image: Image.Image, embedding) -> None:
    """Add condition estimate and 3-category triage to response if classifier is available."""
    cond_obj = _state.get("condition_classifier")
    if cond_obj is None:
        print("Warning: condition_classifier not loaded, skipping triage_categories")
        return

    try:
        from src.data.taxonomy import (
            CONDITION_NAMES, Condition, CONDITION_TRIAGE,
            TriageCategory, TRIAGE_CATEGORY_NAMES,
        )

        # Unpack v2 dict format: {"classifier": clf, "scaler": scaler}
        if isinstance(cond_obj, dict):
            cond_clf = cond_obj["classifier"]
            cond_scaler = cond_obj.get("scaler")
        else:
            cond_clf = cond_obj
            cond_scaler = None

        # Get embedding for condition classifier
        if embedding is not None:
            cond_input = embedding.numpy() if hasattr(embedding, 'numpy') else embedding
        elif _state["e2e_model"] and hasattr(_state["e2e_model"], 'extract_embeddings'):
            emb = _state["e2e_model"].extract_embeddings([image])
            cond_input = emb.cpu().numpy() if emb is not None else None
            if cond_input is None and _state.get("extractor"):
                cond_input = _state["extractor"].extract([image]).numpy()
        elif _state.get("extractor"):
            cond_input = _state["extractor"].extract([image]).numpy()
        else:
            return

        if cond_input is None:
            return

        if cond_scaler is not None:
            cond_input = cond_scaler.transform(cond_input)

        cond_proba = cond_clf.predict_proba(cond_input)
        if cond_proba.ndim != 2:
            return

        # Top-3 individual conditions
        top_indices = np.argsort(cond_proba[0])[::-1][:3]
        condition_probs = []
        for idx in top_indices:
            cond_enum = Condition(int(idx))
            condition_probs.append({
                "condition": CONDITION_NAMES.get(cond_enum, f"Class {idx}"),
                "probability": round(float(cond_proba[0, idx]), 4),
            })

        top_cond = Condition(int(top_indices[0]))
        response["condition_estimate"] = CONDITION_NAMES.get(top_cond, f"Class {top_indices[0]}")
        response["condition_probabilities"] = condition_probs

        # Aggregate into 3 triage categories, anchored to the binary model's
        # malignancy probability (more accurate than the condition classifier).
        # The condition classifier determines the split between inflammatory
        # and benign within the non-malignant portion.
        mal_prob_binary = response["probabilities"]["malignant"]
        non_mal = 1.0 - mal_prob_binary

        # Get raw inflammatory vs benign split from condition classifier
        raw_inflammatory = 0.0
        raw_benign = 0.0
        for idx in range(cond_proba.shape[1]):
            cond_enum = Condition(int(idx))
            cat = CONDITION_TRIAGE.get(cond_enum, TriageCategory.BENIGN)
            p = float(cond_proba[0, idx])
            if cat == TriageCategory.INFLAMMATORY:
                raw_inflammatory += p
            elif cat == TriageCategory.BENIGN:
                raw_benign += p

        # Distribute non-malignant portion proportionally
        raw_non_mal = raw_inflammatory + raw_benign
        if raw_non_mal > 0:
            inflammatory_frac = raw_inflammatory / raw_non_mal
        else:
            inflammatory_frac = 0.0

        category_final = {
            TriageCategory.MALIGNANT: mal_prob_binary,
            TriageCategory.INFLAMMATORY: non_mal * inflammatory_frac,
            TriageCategory.BENIGN: non_mal * (1.0 - inflammatory_frac),
        }

        response["triage_categories"] = {
            cat: {
                "label": TRIAGE_CATEGORY_NAMES[cat],
                "probability": round(prob, 4),
            }
            for cat, prob in category_final.items()
        }
    except Exception as e:
        print(f"Warning: Failed to add condition estimate: {e}")
        import traceback
        traceback.print_exc()


@app.get("/api/health")
async def health():
    model_loaded = _state["e2e_model"] is not None or _state["classifier"] is not None

    if _state["extractor"]:
        device = _state["extractor"].device
    elif _state["e2e_model"]:
        device = _state["e2e_model"].device
    else:
        device = "unknown"

    return {
        "status": "ok",
        "inference_mode": _state["inference_mode"],
        "model_loaded": model_loaded,
        "device": device,
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
