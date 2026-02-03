"""Augmentation pipelines for robustness to imaging conditions."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Dermoscope augmentation module not implemented yet
# from src.data.dermoscope_aug import (
#     get_dermoscope_removal_pipeline,
#     get_dermoscope_addition_pipeline,
# )


def get_dermoscope_removal_pipeline(p: float = 0.5):
    """Placeholder for dermoscope artifact removal."""
    return A.Compose([])


def get_dermoscope_addition_pipeline(p: float = 0.5):
    """Placeholder for dermoscope artifact addition."""
    return A.Compose([])


def get_skin_tone_augmentation(p: float = 0.5):
    """Placeholder for skin tone augmentation."""
    return A.Compose([])


def get_lighting_augmentation():
    """Simulate lighting and exposure variations (different exam rooms, cameras)."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ])


def get_noise_augmentation():
    """Simulate sensor noise (low-light capture, older cameras)."""
    return A.Compose([
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.ISONoise(p=0.3),
    ])


def get_compression_augmentation():
    """Simulate compression artifacts (telemedicine, image uploads)."""
    return A.Compose([
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.5),
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.3),
    ])


def get_domain_bridging_augmentation(source_domain: str, p: float = 0.5):
    """Get domain-bridging augmentation based on source imaging domain.

    Randomly adds or removes dermoscope artifacts so the model cannot rely
    on imaging source as a classification signal.

    Args:
        source_domain: "dermoscopic", "clinical", or "smartphone"
        p: probability of applying the augmentation

    Returns:
        Albumentations Compose pipeline
    """
    if source_domain == "dermoscopic":
        # Remove dermoscope artifacts to look more like phone photos
        return get_dermoscope_removal_pipeline(p=p)
    elif source_domain in ("clinical", "smartphone"):
        # Add dermoscope artifacts to break the domain-pathology correlation
        return get_dermoscope_addition_pipeline(p=p)
    else:
        return A.Compose([])  # No-op for unknown domains


def get_training_transform(image_size: int = 448, domain: str = None):
    """Full training augmentation pipeline for robustness.

    Args:
        image_size: Target image size
        domain: If provided, includes domain-bridging augmentations
    """
    transforms = [
        A.Resize(image_size, image_size),
        # Geometric (orientation-invariant lesions)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
    ]

    # Domain bridging (applied before standard augmentations)
    if domain is not None:
        bridging = get_domain_bridging_augmentation(domain, p=0.4)
        transforms.append(bridging)

    transforms.extend([
        # Lighting variation
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # Sensor noise
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        # Compression artifacts
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        # Normalize for model
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def get_eval_transform(image_size: int = 448):
    """Evaluation transform (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
