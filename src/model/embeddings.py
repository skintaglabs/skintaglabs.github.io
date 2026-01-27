"""MedSigLIP embedding extraction optimized for low-resource environments."""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


class EmbeddingExtractor:
    """Extract embeddings using MedSigLIP vision encoder."""

    def __init__(self, model_name: str = "google/siglip-so400m-patch14-384", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.processor = None

    def load_model(self):
        """Lazy load model to save memory until needed."""
        if self.model is None:
            print(f"Loading model on {self.device}...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
        return self

    def unload_model(self):
        """Free memory after extraction."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def extract(self, images):
        """Extract embeddings from a batch of images.

        Args:
            images: List of PIL images

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        self.load_model()
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        return outputs.cpu()

    @torch.no_grad()
    def extract_dataset(self, images, batch_size: int = 4, cache_path: Path = None):
        """Extract embeddings for a full dataset with batching and caching.

        Args:
            images: List of PIL images
            batch_size: Batch size (use 1-4 for CPU, 8-16 for GPU)
            cache_path: Path to cache embeddings (skips extraction if exists)

        Returns:
            Tensor of shape (num_images, embedding_dim)
        """
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached embeddings from {cache_path}")
            return torch.load(cache_path)

        self.load_model()
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting embeddings"):
            batch = images[i : i + batch_size]
            embeddings = self.extract(batch)
            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(all_embeddings, cache_path)
            print(f"Cached embeddings to {cache_path}")

        return all_embeddings

    @torch.no_grad()
    def extract_text(self, texts):
        """Extract text embeddings for zero-shot classification."""
        self.load_model()
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        return outputs.cpu()


def extract_and_cache(image_dir: Path, cache_path: Path, batch_size: int = 4):
    """Convenience function to extract and cache embeddings from a directory.

    Args:
        image_dir: Directory containing images
        cache_path: Where to save embeddings
        batch_size: Batch size for extraction

    Returns:
        Tuple of (embeddings, image_paths)
    """
    from PIL import Image

    image_paths = sorted(list(Path(image_dir).glob("**/*.jpg")) + list(Path(image_dir).glob("**/*.png")))
    images = [Image.open(p).convert("RGB") for p in tqdm(image_paths, desc="Loading images")]

    extractor = EmbeddingExtractor()
    embeddings = extractor.extract_dataset(images, batch_size=batch_size, cache_path=cache_path)
    extractor.unload_model()

    return embeddings, image_paths
