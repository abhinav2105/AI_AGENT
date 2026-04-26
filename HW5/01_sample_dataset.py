"""
01_sample_dataset.py
Stratified random sampling of the Intel Image Classification dataset.
Copies N_PER_CLASS images from each category to data/sampled/.
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "images" / "seg_train"
SAMPLED_DIR = ROOT / "data" / "sampled"

CATEGORIES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
N_PER_CLASS = 60
SEED = 42


def sample_category(category: str) -> list:
    src = IMAGES_DIR / category
    images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    random.seed(SEED)
    return random.sample(images, min(N_PER_CLASS, len(images)))


def main():
    SAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "output").mkdir(parents=True, exist_ok=True)

    summary = {}
    total = 0

    print("Sampling dataset …")
    for cat in CATEGORIES:
        src_dir = IMAGES_DIR / cat
        all_images = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        sampled = sample_category(cat)

        dest = SAMPLED_DIR / cat
        dest.mkdir(parents=True, exist_ok=True)
        for img in sampled:
            shutil.copy2(img, dest / img.name)

        summary[cat] = {"source_total": len(all_images), "sampled": len(sampled)}
        total += len(sampled)
        print(f"  {cat:<12}  {len(sampled):>3} / {len(all_images)} sampled")

    summary["total"] = total
    out = ROOT / "output" / "sample_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nTotal: {total} images  →  {SAMPLED_DIR.relative_to(ROOT)}")
    print(f"Summary: {out.relative_to(ROOT)}")
    return summary


if __name__ == "__main__":
    main()
