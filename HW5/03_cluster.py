"""
03_cluster.py
KMeans clustering on L2-normalised MobileNetV2 features.
Outputs:
  output/clusters/cluster_assignments.csv
  output/clusters/cluster_metadata.json
  output/clusters/cluster_composition.csv
  output/clusters/elbow_silhouette.png
  output/clusters/pca_scatter.png
  output/clusters/cluster_sizes.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = ROOT / "data" / "features"
CLUSTERS_DIR = ROOT / "output" / "clusters"

CATEGORIES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
N_CLUSTERS = 6
SEED = 42


def load_features():
    data = np.load(FEATURES_DIR / "features.npz", allow_pickle=True)
    return data["features"], data["paths"].tolist(), data["labels"].astype(int)


def elbow_analysis(features: np.ndarray, max_k: int = 12):
    inertias, sil_scores, ks = [], [], range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        km.fit(features)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(features, km.labels_, sample_size=300, random_state=SEED))
        print(f"  k={k:2d}  inertia={km.inertia_:9.1f}  silhouette={sil_scores[-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(ks), inertias, "o-", color="steelblue", linewidth=2)
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Method")
    ax1.grid(alpha=0.3)
    ax2.plot(list(ks), sil_scores, "o-", color="coral", linewidth=2)
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score"); ax2.set_title("Silhouette Analysis")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CLUSTERS_DIR / "elbow_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()


def pca_plot(features: np.ndarray, cluster_labels, true_labels):
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(features)
    var = pca.explained_variance_ratio_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10.colors

    for c in range(N_CLUSTERS):
        m = cluster_labels == c
        ax1.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.7,
                    color=colors[c % 10], label=f"Cluster {c}")
    ax1.set_title(f"KMeans Clusters (k={N_CLUSTERS}) — PCA 2D")
    ax1.legend(markerscale=2, fontsize=8)

    for i, cat in enumerate(CATEGORIES):
        m = true_labels == i
        ax2.scatter(coords[m, 0], coords[m, 1], s=12, alpha=0.7,
                    color=colors[i % 10], label=cat)
    ax2.set_title("True Category Labels — PCA 2D")
    ax2.legend(markerscale=2, fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(CLUSTERS_DIR / "pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)

    features, paths, labels = load_features()
    features_norm = normalize(features)
    print(f"Loaded {len(features)} vectors of dim {features.shape[1]}")

    print("\nElbow + silhouette analysis (k=2…12) …")
    elbow_analysis(features_norm)

    print(f"\nFitting KMeans k={N_CLUSTERS} …")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=15, max_iter=500)
    km.fit(features_norm)
    cluster_labels = km.labels_

    sil = silhouette_score(features_norm, cluster_labels, sample_size=300, random_state=SEED)
    print(f"Silhouette score (k={N_CLUSTERS}): {sil:.4f}")

    df = pd.DataFrame({
        "path": paths,
        "true_label": labels,
        "true_category": [CATEGORIES[l] for l in labels],
        "cluster": cluster_labels,
    })
    df.to_csv(CLUSTERS_DIR / "cluster_assignments.csv", index=False)

    ct = pd.crosstab(df["cluster"], df["true_category"])
    ct.to_csv(CLUSTERS_DIR / "cluster_composition.csv")
    print("\nCluster composition (rows=cluster, cols=category):")
    print(ct.to_string())

    # Cluster-size bar chart
    sizes = df["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(sizes.index, sizes.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("Image Count")
    ax.set_title(f"Images per Cluster (k={N_CLUSTERS})")
    for bar, v in zip(bars, sizes.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, str(v),
                ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(CLUSTERS_DIR / "cluster_sizes.png", dpi=150, bbox_inches="tight")
    plt.close()

    pca_plot(features_norm, cluster_labels, labels)

    # Per-cluster metadata for downstream scripts
    cluster_meta = {}
    for c in range(N_CLUSTERS):
        mask = df["cluster"] == c
        dominant = df[mask]["true_category"].value_counts().idxmax()
        cluster_meta[str(c)] = {
            "size": int(mask.sum()),
            "dominant_category": dominant,
            "image_paths": df[mask]["path"].tolist(),
        }
    (CLUSTERS_DIR / "cluster_metadata.json").write_text(json.dumps(cluster_meta, indent=2))

    print(f"\nCluster metadata → {(CLUSTERS_DIR / 'cluster_metadata.json').relative_to(ROOT)}")
    print(f"PCA scatter      → {(CLUSTERS_DIR / 'pca_scatter.png').relative_to(ROOT)}")
    print(f"Elbow plot       → {(CLUSTERS_DIR / 'elbow_silhouette.png').relative_to(ROOT)}")
    return df, cluster_meta, sil


if __name__ == "__main__":
    main()
