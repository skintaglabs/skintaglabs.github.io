"""Hugging Face model download utilities.

Handles downloading model artifacts from Hugging Face Hub with caching.
"""

import os
from pathlib import Path
from typing import Optional
from huggingface_hub import hf_hub_download


def download_model_from_hf(
    repo_id: str,
    filename: str,
    cache_subdir: str = "models",
    token: Optional[str] = None,
) -> Path:
    """Download model file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., "org/model-name")
        filename: File to download from the repository
        cache_subdir: Subdirectory within HF_HOME for caching
        token: Hugging Face API token (optional, for private repos)

    Returns:
        Path to the downloaded/cached model file

    Environment Variables:
        HF_HOME: Base cache directory (default: ~/.cache/huggingface)
        HF_TOKEN: Hugging Face API token (alternative to token parameter)
    """
    # Get cache directory from env or use default
    hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(hf_home) / cache_subdir

    # Get token from parameter or environment
    token = token or os.getenv("HF_TOKEN")

    print(f"Downloading {repo_id}/{filename} from Hugging Face...")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
        token=token,
    )

    print(f"✓ Model cached at: {model_path}")
    return Path(model_path)


def download_e2e_model_from_hf(
    repo_id: str,
    cache_subdir: str = "models",
    token: Optional[str] = None,
) -> Path:
    """Download end-to-end fine-tuned model directory from Hugging Face Hub.

    Downloads all model files (config.json, model_state.pt, head_state.pt)
    and returns the directory path.

    Args:
        repo_id: Hugging Face repository ID (e.g., "DTanzillo/MedGemma540")
        cache_subdir: Subdirectory within HF_HOME for caching
        token: Hugging Face API token (optional, for private repos)

    Returns:
        Path to the downloaded model directory
    """
    from huggingface_hub import snapshot_download

    # Get cache directory
    hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_dir = Path(hf_home) / cache_subdir

    # Get token
    token = token or os.getenv("HF_TOKEN")

    print(f"Downloading fine-tuned model from {repo_id}...")

    model_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        token=token,
        allow_patterns=["config.json", "model_state.pt", "head_state.pt"],
    )

    print(f"✓ Model downloaded to: {model_dir}")
    return Path(model_dir)


def get_model_config():
    """Get model repository configuration.

    Returns dict with model repository settings from environment variables
    or defaults for SkinTag models.
    """
    return {
        "repo_id": os.getenv("HF_REPO_ID", "DTanzillo/MedGemma540"),
        "classifier_filename": os.getenv("HF_CLASSIFIER_FILE", "classifier_deep_mlp.pkl"),
        "condition_classifier_filename": os.getenv("HF_CONDITION_FILE", "classifier_condition.pkl"),
    }
