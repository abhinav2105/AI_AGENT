"""
02_extract_features.py
Extracts 1280-dim MobileNetV2 feature vectors for every sampled image.
Saves data/features/features.npz  (features, paths, labels)
      data/features/metadata.json
"""

import json
import os
import numpy as np
from pathlib import Path
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

ROOT = Path(__file__).resolve().parent.parent
SAMPLED_DIR = ROOT / "data" / "sampled"
FEATURES_DIR = ROOT / "data" / "features"

CATEGORIES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model():
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    out = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=out)
    model.trainable = False
    return model, preprocess_input


def load_image_pil(path: str, preprocess_fn) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return preprocess_fn(arr[np.newaxis])[0]


def collect_paths():
    paths, labels = [], []
    label_map = {cat: i for i, cat in enumerate(CATEGORIES)}
    for cat in CATEGORIES:
        cat_dir = SAMPLED_DIR / cat
        imgs = sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.png"))
        paths.extend([str(p) for p in imgs])
        labels.extend([label_map[cat]] * len(imgs))
    return paths, labels, label_map


def extract_all(paths, model, preprocess_fn) -> np.ndarray:
    all_feats = []
    n = len(paths)
    for i in range(0, n, BATCH_SIZE):
        batch = paths[i : i + BATCH_SIZE]
        imgs = np.stack([load_image_pil(p, preprocess_fn) for p in batch])
        feats = model.predict(imgs, verbose=0)
        all_feats.append(feats)
        print(f"  Extracted {min(i + BATCH_SIZE, n):>3}/{n}", end="\r")
    print()
    return np.vstack(all_feats)


def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    paths, labels, label_map = collect_paths()
    print(f"Found {len(paths)} sampled images across {len(CATEGORIES)} categories")

    model, preprocess_fn = build_model()
    print("MobileNetV2 loaded (ImageNet weights, no top). Extracting features …")

    features = extract_all(paths, model, preprocess_fn)

    np.savez_compressed(
        FEATURES_DIR / "features.npz",
        features=features,
        paths=np.array(paths),
        labels=np.array(labels),
    )
    meta = {
        "n_images": len(paths),
        "feature_dim": int(features.shape[1]),
        "categories": CATEGORIES,
        "label_map": label_map,
    }
    (FEATURES_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Feature matrix: {features.shape}  (dtype={features.dtype})")
    print(f"Saved → {(FEATURES_DIR / 'features.npz').relative_to(ROOT)}")
    return features, paths, labels


if __name__ == "__main__":
    main()
