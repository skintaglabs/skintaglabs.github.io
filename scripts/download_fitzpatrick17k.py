"""Download Fitzpatrick17k images from URLs in the CSV."""
import os
import sys
import time
import hashlib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CSV_PATH = Path("data/fitzpatrick17k/fitzpatrick17k.csv")
IMG_DIR = Path("data/fitzpatrick17k/images")
MAX_WORKERS = 16
TIMEOUT = 15
RETRY = 2


def download_one(row):
    md5hash = row["md5hash"]
    url = row["url"]
    if pd.isna(url) or not url.strip():
        return md5hash, "skip_null"

    ext = Path(url).suffix.split("?")[0] or ".jpg"
    dest = IMG_DIR / f"{md5hash}{ext}"
    if dest.exists():
        return md5hash, "exists"

    for attempt in range(RETRY + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.content) > 500:
                dest.write_bytes(r.content)
                return md5hash, "ok"
            else:
                last_err = f"status={r.status_code} size={len(r.content)}"
        except Exception as e:
            last_err = str(e)
        if attempt < RETRY:
            time.sleep(1)

    return md5hash, f"fail: {last_err}"


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    rows = df.to_dict("records")
    total = len(rows)

    counts = {"ok": 0, "exists": 0, "fail": 0, "skip_null": 0}
    failed = []

    print(f"Downloading {total} images with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, r): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            md5, status = fut.result()
            if status.startswith("fail"):
                counts["fail"] += 1
                failed.append((md5, status))
            else:
                counts[status] += 1
            if i % 200 == 0 or i == total:
                print(f"  [{i}/{total}] ok={counts['ok']} exists={counts['exists']} fail={counts['fail']} skip={counts['skip_null']}")

    print(f"\nDone. ok={counts['ok']} exists={counts['exists']} fail={counts['fail']} skip={counts['skip_null']}")
    if failed:
        print(f"\nFailed downloads ({len(failed)}):")
        for md5, err in failed[:20]:
            print(f"  {md5}: {err}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
