"""Hugging Face model download utilities.

Handles downloading model artifacts from Hugging Face Hub with caching.
"""

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, snapshot_download


def _get_cache_dir(cache_subdir: str) -> Path:
    """Get cache directory from environment or default."""
    hf_home = os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / cache_subdir


def _get_token(token: Optional[str]) -> Optional[str]:
    """Get token from parameter or environment."""
    return token or os.getenv("HF_TOKEN")


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
    """
    print(f"Downloading {repo_id}/{filename} from Hugging Face...")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(_get_cache_dir(cache_subdir)),
        token=_get_token(token),
    )

    print(f"Model cached at: {model_path}")
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
        repo_id: Hugging Face repository ID (e.g., "skintaglabs/siglip-skin-lesion-classifier")
        cache_subdir: Subdirectory within HF_HOME for caching
        token: Hugging Face API token (optional, for private repos)

    Returns:
        Path to the downloaded model directory
    """
    print(f"Downloading fine-tuned model from {repo_id}...")

    model_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(_get_cache_dir(cache_subdir)),
        token=_get_token(token),
        allow_patterns=["config.json", "model_state.pt", "head_state.pt"],
    )

    print(f"Model downloaded to: {model_dir}")
    return Path(model_dir)


def get_model_config():
    """Get model repository configuration.

    Returns dict with model repository settings from environment variables
    or defaults for SkinTag models.
    """
    return {
        "repo_id": os.getenv("HF_REPO_ID", "skintaglabs/siglip-skin-lesion-classifier"),
        "classifier_filename": os.getenv("HF_CLASSIFIER_FILE", "Misc/classifier_deep_mlp.pkl"),
        "condition_classifier_filename": os.getenv("HF_CONDITION_FILE", "Misc/classifier_condition.pkl"),
    }
