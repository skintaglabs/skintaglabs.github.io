"""Dataset loading utilities for skin lesion datasets."""

# Development notes:
# - Developed with AI assistance (Claude/Anthropic) for implementation and refinement
# - Code simplified using Anthropic's code-simplifier agent (https://www.anthropic.com/claude-code)
# - Core architecture and domain logic by SkinTag team

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

from src.data.schema import SkinSample, samples_to_arrays


# HAM10000 class mapping
CLASS_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

# Binary mapping for simplified classification
BINARY_MAPPING = {
    "akiec": 1,  # malignant
    "bcc": 1,    # malignant
    "bkl": 0,    # benign
    "df": 0,     # benign
    "mel": 1,    # malignant
    "nv": 0,     # benign
    "vasc": 0,   # benign
}


def load_ham10000(data_dir: Path, binary: bool = True):
    """Load HAM10000 dataset from Kaggle download.

    Expected structure after unzipping:
        data_dir/
            HAM10000_metadata.csv
            HAM10000_images_part_1/
            HAM10000_images_part_2/
        OR
            Skin Cancer/  (Kaggle folder name)
                HAM10000_metadata.csv
                ...

    Args:
        data_dir: Path to data directory
        binary: If True, map to benign (0) vs malignant (1)

    Returns:
        Tuple of (images, labels, metadata_df)
    """
    data_dir = Path(data_dir)

    # Find metadata file
    metadata_candidates = [
        data_dir / "HAM10000_metadata.csv",
        data_dir / "Skin Cancer" / "HAM10000_metadata.csv",
    ]
    metadata_path = next((p for p in metadata_candidates if p.exists()), None)
    if metadata_path is None:
        raise FileNotFoundError(f"HAM10000_metadata.csv not found in {data_dir}")

    # Find image directories
    image_dirs = []
    for pattern in ["HAM10000_images_part_*", "Skin Cancer"]:
        image_dirs.extend(data_dir.glob(pattern))
    if not image_dirs:
        image_dirs = [data_dir]

    # Build image path lookup
    image_lookup = {}
    for img_dir in image_dirs:
        for img_path in img_dir.glob("**/*.jpg"):
            image_lookup[img_path.stem] = img_path

    # Load metadata
    df = pd.read_csv(metadata_path)

    images = []
    labels = []
    valid_indices = []

    for idx, row in df.iterrows():
        image_id = row["image_id"]
        if image_id not in image_lookup:
            continue

        img = Image.open(image_lookup[image_id]).convert("RGB")
        images.append(img)

        if binary:
            labels.append(BINARY_MAPPING[row["dx"]])
        else:
            labels.append(list(CLASS_NAMES.keys()).index(row["dx"]))

        valid_indices.append(idx)

    metadata = df.iloc[valid_indices].reset_index(drop=True)

    return images, labels, metadata


def load_multi_dataset(
    data_dir: Path,
    datasets: list[str] = None,
    dataset_options: dict = None,
) -> list[SkinSample]:
    """Load and merge multiple datasets into unified SkinSample list.

    Args:
        data_dir: Root data directory
        datasets: List of dataset names to load (default: all available)
        dataset_options: Per-dataset options dict, e.g. {"fitzpatrick17k": {"exclude_uncertain": True}}

    Returns:
        List of SkinSample from all requested datasets
    """
    from src.data.datasets import DATASET_LOADERS

    if datasets is None:
        datasets = list(DATASET_LOADERS.keys())
    if dataset_options is None:
        dataset_options = {}

    all_samples = []
    for name in datasets:
        if name not in DATASET_LOADERS:
            print(f"Warning: Unknown dataset '{name}', skipping")
            continue

        loader = DATASET_LOADERS[name]
        opts = dataset_options.get(name, {})

        try:
            samples = loader(Path(data_dir), **opts)
            print(f"Loaded {len(samples)} samples from {name}")
            all_samples.extend(samples)
        except FileNotFoundError as e:
            print(f"Warning: Could not load {name}: {e}")
            continue

    print(f"Total: {len(all_samples)} samples from {len(datasets)} datasets")
    return all_samples


def get_demographic_groups(metadata: pd.DataFrame):
    """Extract demographic groups for fairness analysis.

    Supports both legacy HAM10000 metadata and unified multi-dataset metadata.

    Returns:
        Dictionary with group assignments for each sample
    """
    groups = {}

    # Age groups
    if "age" in metadata.columns:
        age_bins = [0, 30, 50, 70, 100]
        age_labels = ["<30", "30-50", "50-70", "70+"]
        groups["age"] = pd.cut(
            metadata["age"].fillna(50), bins=age_bins, labels=age_labels
        ).astype(str).values

    # Sex
    if "sex" in metadata.columns:
        groups["sex"] = metadata["sex"].fillna("unknown").values

    # Body location
    loc_col = "localization" if "localization" in metadata.columns else "location"
    if loc_col in metadata.columns:
        groups["location"] = metadata[loc_col].fillna("unknown").values

    # Fitzpatrick skin type (new: multi-dataset)
    if "fitzpatrick" in metadata.columns:
        fitz = metadata["fitzpatrick"].copy()
        # Normalize to clean strings: 2.0 -> "2", NaN -> "unknown"
        def _norm_fitz(v):
            try:
                return str(int(float(v)))
            except (ValueError, TypeError):
                return "unknown"
        groups["fitzpatrick"] = np.array([_norm_fitz(v) for v in fitz])

    # Domain (new: multi-dataset)
    if "domain" in metadata.columns:
        groups["domain"] = metadata["domain"].fillna("unknown").values

    # Dataset source (new: multi-dataset)
    if "dataset" in metadata.columns:
        groups["dataset"] = metadata["dataset"].fillna("unknown").values

    return groups
