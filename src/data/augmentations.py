"""Augmentation pipelines for skin lesion robustness."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_skin_tone_augmentation():
    """Simulate skin tone variations via color space shifts."""
    return A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.7
        ),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
    ])


def get_lighting_augmentation():
    """Simulate lighting and exposure variations."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ])


def get_noise_augmentation():
    """Simulate image noise and quality degradation."""
    return A.Compose([
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.ISONoise(p=0.3),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
    ])


def get_training_transform(image_size: int = 448):
    """Full training augmentation pipeline."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        # Skin tone
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        # Lighting
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        # Noise
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_eval_transform(image_size: int = 448):
    """Evaluation transform (no augmentation)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
