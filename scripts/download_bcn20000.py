"""Download BCN20000 dataset images and metadata from the ISIC Archive.

Uses ISIC Archive API v2 search endpoint to paginate through collection 249
(BCN20000), download images, and save metadata as CSV.
"""
import os
import sys
import csv
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://api.isic-archive.com/api/v2"
COLLECTION_ID = 249
IMG_DIR = Path("data/bcn20000/images")
META_CSV = Path("data/bcn20000/bcn20000_metadata.csv")
MAX_WORKERS = 16
PAGE_LIMIT = 100
TIMEOUT = 30
RETRY = 3


def fetch_all_image_metadata():
    """Paginate through the ISIC search API using cursor-based pagination."""
    all_images = []
    url = f"{BASE_URL}/images/search/?collections={COLLECTION_ID}&limit={PAGE_LIMIT}"
    page = 0

    while url:
        page += 1
        print(f"  Fetching metadata page {page} ({len(all_images)} so far) ...")
        for attempt in range(RETRY + 1):
            try:
                r = requests.get(url, timeout=TIMEOUT)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == RETRY:
                    print(f"  FATAL: Could not fetch metadata page {page}: {e}")
                    return all_images
                print(f"  Retry {attempt+1}/{RETRY}: {e}")
                time.sleep(2 ** attempt)

        results = data.get("results", [])
        if not results:
            break

        all_images.extend(results)
        url = data.get("next")  # cursor-based next URL

    return all_images


def extract_metadata_row(img):
    """Extract a flat metadata dict from an ISIC image object."""
    isic_id = img.get("isic_id", "")
    meta = img.get("metadata", {}) or {}
    clinical = meta.get("clinical", {}) or {}

    # BCN20000 uses diagnosis_1 through diagnosis_5 hierarchy
    # diagnosis_1 = Benign/Malignant, diagnosis_3 = specific diagnosis
    diagnosis = clinical.get("diagnosis_3") or clinical.get("diagnosis_2") or ""
    benign_malignant = clinical.get("diagnosis_1", "")

    return {
        "isic_id": isic_id,
        "diagnosis": diagnosis,
        "benign_malignant": benign_malignant,
    }


def save_metadata_csv(images):
    """Save metadata CSV from the list of image objects."""
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = [extract_metadata_row(img) for img in images]
    with open(META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["isic_id", "diagnosis", "benign_malignant"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved metadata for {len(rows)} images to {META_CSV}")
    return rows


def download_one(isic_id):
    """Download a single image. Uses the full-size URL from S3 (already 1024x1024 for BCN20000)."""
    dest = IMG_DIR / f"{isic_id}.jpg"
    if dest.exists() and dest.stat().st_size > 500:
        return isic_id, "exists"

    # Use the direct S3 URL pattern for faster downloads
    url = f"https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg"
    for attempt in range(RETRY + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200 and len(r.content) > 500:
                dest.write_bytes(r.content)
                return isic_id, "ok"
            else:
                last_err = f"status={r.status_code} size={len(r.content)}"
        except Exception as e:
            last_err = str(e)
        if attempt < RETRY:
            time.sleep(1 * (attempt + 1))

    return isic_id, f"fail: {last_err}"


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch all metadata
    print("=== Step 1: Fetching image metadata from ISIC Archive ===")
    images = fetch_all_image_metadata()
    if not images:
        print("ERROR: No images found in collection. Exiting.")
        sys.exit(1)
    print(f"Total images in collection: {len(images)}\n")

    # Step 2: Save metadata CSV
    print("=== Step 2: Saving metadata CSV ===")
    rows = save_metadata_csv(images)
    print()

    # Step 3: Download images
    isic_ids = [r["isic_id"] for r in rows]
    print(f"=== Step 3: Downloading {len(isic_ids)} images with {MAX_WORKERS} workers ===")

    counts = {"ok": 0, "exists": 0, "fail": 0}
    failed = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, iid): iid for iid in isic_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            iid, status = fut.result()
            if status.startswith("fail"):
                counts["fail"] += 1
                failed.append((iid, status))
            else:
                counts[status] += 1
            if i % 200 == 0 or i == len(isic_ids):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                print(
                    f"  [{i}/{len(isic_ids)}] "
                    f"ok={counts['ok']} exists={counts['exists']} fail={counts['fail']} "
                    f"({rate:.1f} img/s, {elapsed:.0f}s elapsed)"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. ok={counts['ok']} exists={counts['exists']} fail={counts['fail']}")
    if failed:
        print(f"\nFailed downloads ({len(failed)}):")
        for iid, err in failed[:30]:
            print(f"  {iid}: {err}")
        if len(failed) > 30:
            print(f"  ... and {len(failed) - 30} more")


if __name__ == "__main__":
    main()
