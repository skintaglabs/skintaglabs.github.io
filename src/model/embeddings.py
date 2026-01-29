"""MedSigLIP embedding extraction optimized for low-resource environments."""

import hashlib
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor


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
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
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
        vision_model = getattr(self.model, "vision_model", self.model)
        outputs = vision_model(**inputs)
        # Use pooler_output if available, else mean-pool last_hidden_state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output.cpu()
        return outputs.last_hidden_state.mean(dim=1).cpu()

    @staticmethod
    def _load_image(item):
        """Load a single image from a path string or return a PIL Image as-is."""
        from PIL import Image as _Image
        if isinstance(item, str):
            return _Image.open(item).convert("RGB")
        if isinstance(item, Path):
            return _Image.open(str(item)).convert("RGB")
        return item  # already a PIL Image

    @torch.no_grad()
    def extract_dataset(
        self,
        images,
        batch_size: int = 4,
        cache_path: Path = None,
        transform=None,
        augmentation_config: dict = None,
    ):
        """Extract embeddings for a full dataset with batching and caching.

        Images are loaded lazily per-batch to avoid holding all PIL objects
        in RAM simultaneously.

        Args:
            images: List of PIL images OR list of file path strings
            batch_size: Batch size (use 1-4 for CPU, 8-16 for GPU)
            cache_path: Path to cache embeddings (skips extraction if exists)
            transform: Optional augmentation transform (applied per-image before extraction)
            augmentation_config: If provided, hashed into cache filename to avoid stale caches

        Returns:
            Tensor of shape (num_images, embedding_dim)
        """
        # Build cache path with augmentation hash
        effective_cache = cache_path
        if cache_path and augmentation_config:
            config_hash = hashlib.md5(
                json.dumps(augmentation_config, sort_keys=True).encode()
            ).hexdigest()[:8]
            stem = Path(cache_path).stem
            effective_cache = Path(cache_path).parent / f"{stem}_aug{config_hash}.pt"

        if effective_cache and Path(effective_cache).exists():
            print(f"Loading cached embeddings from {effective_cache}")
            return torch.load(effective_cache)

        self.load_model()
        all_embeddings = []

        for i in tqdm(range(0, len(images), batch_size), desc="Extracting embeddings"):
            batch_items = images[i : i + batch_size]
            # Lazy load: convert paths to PIL images per-batch
            batch = [self._load_image(item) for item in batch_items]

            if transform is not None:
                augmented = []
                for img in batch:
                    arr = np.array(img)
                    aug_arr = transform(image=arr)["image"]
                    # If transform returns tensor (has ToTensorV2), convert back to PIL
                    if isinstance(aug_arr, (torch.Tensor, np.ndarray)):
                        if isinstance(aug_arr, torch.Tensor):
                            aug_arr = aug_arr.numpy()
                        if aug_arr.ndim == 3 and aug_arr.shape[0] == 3:
                            aug_arr = aug_arr.transpose(1, 2, 0)
                        # Denormalize if normalized
                        if aug_arr.max() <= 1.0:
                            aug_arr = (aug_arr * 255).clip(0, 255).astype(np.uint8)
                        from PIL import Image
                        augmented.append(Image.fromarray(aug_arr))
                    else:
                        from PIL import Image
                        augmented.append(Image.fromarray(aug_arr))
                batch = augmented

            embeddings = self.extract(batch)
            all_embeddings.append(embeddings)
            # Free batch images immediately (important for path-based loading)
            del batch

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if effective_cache:
            Path(effective_cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save(all_embeddings, effective_cache)
            print(f"Cached embeddings to {effective_cache}")

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
