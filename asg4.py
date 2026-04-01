
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. LOAD DATA
# =============================================================================
df = pd.read_csv("/Users/abhi/umbc/seng691/asg4/merged_decibel_data.csv")
df.columns = ["dBA", "Timestamp", "Location"]
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["dBA"] = pd.to_numeric(df["dBA"], errors="coerce")

# Enforce display order
LOCATION_ORDER = ["RAC", "UC", "Library", "Commons"]
PALETTE = {"RAC": "#E63946", "UC": "#457B9D", "Library": "#2A9D8F", "Commons": "#E9C46A"}

# =============================================================================
# 2. DATA QUALITY CHECK
# =============================================================================
print("=" * 50)
print("DATA QUALITY CHECK")
print("=" * 50)
print(f"Total rows        : {len(df)}")
print(f"\nMissing values:")
print(df.isnull().sum().to_string())
print(f"\nDuplicate rows    : {df.duplicated().sum()}")
print(f"dBA range         : {df['dBA'].min():.1f} – {df['dBA'].max():.1f} dBA")
print(f"\nData points per location:")
print(df["Location"].value_counts().to_string())

# =============================================================================
# 3. SUMMARY STATISTICS BY LOCATION
# =============================================================================
print("\n" + "=" * 50)
print("SUMMARY STATISTICS BY LOCATION")
print("=" * 50)
stats = (
    df.groupby("Location")["dBA"]
    .agg(Count="count", Mean="mean", Median="median", Std="std", Min="min", Max="max")
    .reindex(LOCATION_ORDER)
    .round(2)
)
print(stats.to_string())

# =============================================================================
# 4. PATTERN OBSERVATIONS
# =============================================================================
print("\n" + "=" * 50)
print("KEY OBSERVATIONS")
print("=" * 50)
loudest  = stats["Mean"].idxmax()
quietest = stats["Mean"].idxmin()
print(f"  Loudest location  : {loudest}  ({stats.loc[loudest, 'Mean']} dBA avg)")
print(f"  Quietest location : {quietest} ({stats.loc[quietest, 'Mean']} dBA avg)")
print(f"  Most variable     : {stats['Std'].idxmax()} (Std = {stats['Std'].max()} dBA)")
print(f"  Most consistent   : {stats['Std'].idxmin()} (Std = {stats['Std'].min()} dBA)")

# =============================================================================
# 5. PLOT 1 – Box + Strip Plot  (Distribution per location)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))

sns.boxplot(
    data=df, x="Location", y="dBA",
    order=LOCATION_ORDER,
    hue="Location", palette=PALETTE,
    width=0.45, linewidth=1.4, legend=False,
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
    ax=ax,
)
sns.stripplot(
    data=df, x="Location", y="dBA",
    order=LOCATION_ORDER,
    hue="Location", palette=PALETTE,
    alpha=0.25, size=3, jitter=True, legend=False,
    ax=ax,
)

ax.set_title("Sound Level Distribution by Location", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Location", fontsize=11)
ax.set_ylabel("Sound Level (dBA)", fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("plot1_boxplot.png", dpi=150)
plt.close()
print("\nSaved: plot1_boxplot.png")

# =============================================================================
# 6. PLOT 2 – Mean ± Std Bar Chart
# =============================================================================
means  = [stats.loc[loc, "Mean"] for loc in LOCATION_ORDER]
stds   = [stats.loc[loc, "Std"]  for loc in LOCATION_ORDER]
colors = [PALETTE[loc]            for loc in LOCATION_ORDER]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    LOCATION_ORDER, means, yerr=stds, capsize=6,
    color=colors, edgecolor="black", linewidth=0.8,
    width=0.5, error_kw=dict(elinewidth=1.4),
)
for bar, m, s in zip(bars, means, stds):
    ax.text(
        bar.get_x() + bar.get_width() / 2, m + s + 0.5,
        f"{m:.1f} dBA", ha="center", va="bottom",
        fontsize=9.5, fontweight="bold",
    )

ax.set_title("Mean Sound Level (± Std Dev) by Location", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Location", fontsize=11)
ax.set_ylabel("Mean Sound Level (dBA)", fontsize=11)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.set_ylim(0, max(means) + max(stds) + 5)
plt.tight_layout()
plt.savefig("plot2_mean_bar.png", dpi=150)
plt.close()
print("Saved: plot2_mean_bar.png")

# =============================================================================
# 7. PLOT 3 – KDE Density Curves
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
for loc in LOCATION_ORDER:
    subset = df[df["Location"] == loc]["dBA"]
    subset.plot.kde(ax=ax, label=loc, color=PALETTE[loc], linewidth=2.2)
    ax.axvline(subset.mean(), color=PALETTE[loc], linestyle="--", linewidth=1.2, alpha=0.7)

ax.set_title("Sound Level Density Distribution by Location", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Sound Level (dBA)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.legend(title="Location", fontsize=10)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("plot3_kde.png", dpi=150)
plt.close()
print("Saved: plot3_kde.png")

print("\nPart A analysis complete.")