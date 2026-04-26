# HW5 – Neural Networks and Deep Learning
**SENG 691 AI Agent Computing**

## Project Overview

This project implements an end-to-end image analysis pipeline on the Intel Image Classification dataset. It samples 360 images across six scene categories (buildings, forest, glacier, mountain, sea, street), extracts 1,280-dimensional feature vectors using a pre-trained MobileNetV2 CNN, groups them into six clusters with KMeans, and produces slideshow-style MP4 videos for each cluster. Background music is added algorithmically: per-cluster HSV colour statistics are mapped to musical parameters (tempo, mode, root pitch) and a WAV track is synthesised from scratch using NumPy, then embedded in the final video — no manual or random music selection is involved.

---

## Folder Structure

```
asg_5/
├── requirements.txt
├── README.md
├── SENG 691 - HW5.pdf
│
├── data/
│   ├── images/
│   │   └── seg_train/          # original Intel dataset (not included in submission)
│   │       ├── buildings/
│   │       ├── forest/
│   │       ├── glacier/
│   │       ├── mountain/
│   │       ├── sea/
│   │       └── street/
│   ├── sampled/                # created by 01_sample_dataset.py
│   │   ├── buildings/          # 60 images per category
│   │   ├── forest/
│   │   ├── glacier/
│   │   ├── mountain/
│   │   ├── sea/
│   │   └── street/
│   └── features/               # created by 02_extract_features.py
│       ├── features.npz        # (360, 1280) float32 feature matrix
│       └── metadata.json
│
├── output/
│   ├── sample_summary.json     # created by 01
│   ├── music_selection_report.json  # created by 05
│   ├── clusters/               # created by 03_cluster.py
│   │   ├── cluster_assignments.csv
│   │   ├── cluster_composition.csv
│   │   ├── cluster_metadata.json
│   │   ├── elbow_silhouette.png
│   │   ├── pca_scatter.png
│   │   └── cluster_sizes.png
│   ├── videos/                 # created by 04 and 05
│   │   ├── cluster_0_silent.mp4
│   │   ├── cluster_0_final.mp4
│   │   └── ...                 # repeated for clusters 1–5
│   └── audio/                  # created by 05_music_selector.py
│       ├── cluster_0_music.wav
│       └── ...                 # repeated for clusters 1–5
│
└── scripts/
    ├── 01_sample_dataset.py
    ├── 02_extract_features.py
    ├── 03_cluster.py
    ├── 04_generate_video.py
    ├── 05_music_selector.py
    └── HW5_Notebook.ipynb
```

---

## Setup

**Requirements:** Python 3.10 or later.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon (M-series) note:** the standard `tensorflow` package includes
> Metal acceleration support. No additional packages are required.

---

## How to Run

### Option A — Jupyter Notebook (recommended)

Open `scripts/HW5_Notebook.ipynb` from the **project root** directory.  
The setup cell sets `ROOT = Path().resolve()`, so the notebook must be opened
with the working directory set to the project root (the default when launching
Jupyter from that directory).

```bash
jupyter notebook scripts/HW5_Notebook.ipynb
```

Run all cells in order. Each `%run` cell executes the corresponding script and
the following cells display its outputs inline.

### Option B — Run scripts individually from the project root

All five scripts must be run from the project root so that relative paths
(`data/`, `output/`) resolve correctly.

```bash
python scripts/01_sample_dataset.py    # ~5 s
python scripts/02_extract_features.py  # ~2–5 min (downloads MobileNetV2 weights on first run)
python scripts/03_cluster.py           # ~1 min
python scripts/04_generate_video.py    # ~3–5 min
python scripts/05_music_selector.py    # ~2–3 min
```

---

## Expected Outputs

| Step | Output |
|------|--------|
| 01 | `data/sampled/<category>/` — 60 JPEG images per category (360 total) |
| 01 | `output/sample_summary.json` — per-category counts |
| 02 | `data/features/features.npz` — (360, 1280) float32 feature matrix |
| 02 | `data/features/metadata.json` — category names and label map |
| 03 | `output/clusters/cluster_assignments.csv` — image path, true label, cluster ID |
| 03 | `output/clusters/cluster_composition.csv` — cross-tab of cluster vs. category |
| 03 | `output/clusters/cluster_metadata.json` — per-cluster image lists |
| 03 | `output/clusters/elbow_silhouette.png` — inertia and silhouette vs. k |
| 03 | `output/clusters/pca_scatter.png` — 2D PCA projection of clusters and true labels |
| 03 | `output/clusters/cluster_sizes.png` — bar chart of images per cluster |
| 04 | `output/videos/cluster_<id>_silent.mp4` — one silent slideshow per cluster |
| 05 | `output/audio/cluster_<id>_music.wav` — synthesised audio track per cluster |
| 05 | `output/videos/cluster_<id>_final.mp4` — final video with embedded music |
| 05 | `output/music_selection_report.json` — colour profile and music parameters per cluster |

---

## Dataset

**Intel Image Classification**  
Source: Kaggle — https://www.kaggle.com/datasets/puneet6060/intel-image-classification  
License: Open (CC0 / public domain)  
~14,000 training images across 6 scene categories at 150×150 px JPEG.

Place the downloaded archive contents so that the six category folders sit at:
```
data/images/seg_train/buildings/
data/images/seg_train/forest/
...
```
