"""
SENG 691 – Assignment 4
Part B: Applying Machine Learning Algorithms
---------------------------------------------
Target   : Location (RAC / UC / Library / Commons)  [Multi-class]
Features : 3 clean engineered features (no data leakage)
             1. dBA              – raw sound level
             2. Rolling_Std      – local variability (window=5)
             3. Rolling_Mean     – local moving average (window=5)
Models   : Decision Tree, Random Forest, Support Vector Machine (SVM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv("/Users/abhi/umbc/seng691/asg4/merged_decibel_data.csv")
df.columns = ["dBA", "Timestamp", "Location"]
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["dBA"] = pd.to_numeric(df["dBA"], errors="coerce")
df.dropna(inplace=True)

# =============================================================================
# 2. FEATURE ENGINEERING  (3 clean features — no leakage)
# =============================================================================

# Feature 1: Raw dBA — the actual sound level reading
# (already present, no transformation needed)

# Feature 2: Rolling Standard Deviation (window = 5 readings per location)
# Captures how variable / bursty the sound is locally
df["Rolling_Std"] = (
    df.groupby("Location")["dBA"]
    .transform(lambda x: x.rolling(window=5, min_periods=1).std().fillna(0))
)

# Feature 3: Rolling Mean (window = 5 readings per location)
# Captures the local average sound level — smooths out spikes
df["Rolling_Mean"] = (
    df.groupby("Location")["dBA"]
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)

print("=" * 55)
print("ENGINEERED FEATURES — SAMPLE")
print("=" * 55)
print(df[["dBA", "Rolling_Std", "Rolling_Mean", "Location"]].head(12).to_string())

print("\nData points per location:")
print(df["Location"].value_counts().reindex(["RAC","UC","Library","Commons"]).to_string())

# =============================================================================
# 3. PREPARE DATA FOR ML
# =============================================================================
FEATURES     = ["dBA", "Rolling_Std", "Rolling_Mean"]
TARGET       = "Location"
LOCATION_ORDER = ["RAC", "UC", "Library", "Commons"]

X = df[FEATURES]
y = df[TARGET]

# Encode target labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
CLASS_NAMES = le.classes_   # ['Commons', 'Library', 'RAC', 'UC']

# Stratified split — preserves class balance in train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"\nTrain size : {len(X_train)} samples")
print(f"Test size  : {len(X_test)}  samples")
print(f"Classes    : {list(CLASS_NAMES)}")

# =============================================================================
# 4. TRAIN 3 MODELS
# =============================================================================
models = {
    "Decision Tree" : DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest" : RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM"           : SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
}

results = {}
cms     = {}
reports = {}

print("\n" + "=" * 55)
print("MODEL TRAINING & EVALUATION")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    cms[name]     = confusion_matrix(y_test, y_pred)
    reports[name] = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print(f"\n{name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Classification Report:\n{reports[name]}")

# =============================================================================
# 5. PLOT A — Model Performance Comparison Bar Chart
# =============================================================================
metrics_df  = pd.DataFrame(results).T
metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
METRIC_COLORS = ["#457B9D", "#2A9D8F", "#E9C46A", "#E63946"]

fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(metrics_df))
width = 0.2

for i, (metric, color) in enumerate(zip(metric_cols, METRIC_COLORS)):
    bars = ax.bar(x + i * width, metrics_df[metric], width,
                  label=metric, color=color, edgecolor="black", linewidth=0.7)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_df.index, fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Performance Comparison — Location Classification",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("plot4_model_comparison.png", dpi=150)
plt.close()
print("\nSaved: plot4_model_comparison.png")

# =============================================================================
# 6. PLOT B — Confusion Matrices (3 subplots)
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cm) in zip(axes, cms.items()):
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax, cbar=False,
    )
    acc_val = results[name]["Accuracy"]
    ax.set_title(f"{name}\n(Acc: {acc_val:.2f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)

plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plot5_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: plot5_confusion_matrices.png")

# =============================================================================
# 7. PLOT C — Random Forest Feature Importance
# =============================================================================
rf_model    = models["Random Forest"]
importances = rf_model.feature_importances_
feat_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#E63946" if v == feat_series.max() else "#457B9D" for v in feat_series]
feat_series.plot.barh(ax=ax, color=colors, edgecolor="black", linewidth=0.7)
for i, (val, name) in enumerate(zip(feat_series.values, feat_series.index)):
    ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=10)
ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Importance Score", fontsize=11)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
highlight = mpatches.Patch(color="#E63946", label="Most important feature")
ax.legend(handles=[highlight], fontsize=9)
plt.tight_layout()
plt.savefig("plot6_feature_importance.png", dpi=150)
plt.close()
print("Saved: plot6_feature_importance.png")

# =============================================================================
# 8. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
best_model = max(results, key=lambda m: results[m]["F1-Score"])
print(f"Best performing model : {best_model}")
print(f"  F1-Score  : {results[best_model]['F1-Score']:.4f}")
print(f"  Accuracy  : {results[best_model]['Accuracy']:.4f}")
print("\nAll model scores:")
print(f"  {'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("  " + "-" * 52)
for name, r in results.items():
    print(f"  {name:<18} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f} {r['Recall']:>10.4f} {r['F1-Score']:>10.4f}")

print("\nPart B complete.")