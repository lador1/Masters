from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore[import]

RESULTS_DIR = Path(r"C:\Masters\results\mmfit")
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Existing result files
basic_summary_path = RESULTS_DIR / "pose_baseline_summary.csv"
v2_summary_path = RESULTS_DIR / "pose_baseline_v2_summary.csv"
v2_confusion_path = RESULTS_DIR / "pose_baseline_v2_confusion_matrix.csv"
v2_importance_path = RESULTS_DIR / "pose_baseline_v2_feature_importance.csv"

# -----------------------------
# 1. Baseline comparison chart
# -----------------------------
basic = pd.read_csv(basic_summary_path)
v2 = pd.read_csv(v2_summary_path)

comparison = pd.concat([basic, v2], ignore_index=True)

comparison["label"] = comparison["modality"]

comparison_plot = comparison.set_index("label")[["accuracy", "macro_f1"]]

ax = comparison_plot.plot(kind="bar", figsize=(8, 5))
ax.set_title("Pose-only baseline comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xlabel("Model")
plt.xticks(rotation=0)
plt.tight_layout()

comparison_fig = FIGURE_DIR / "pose_baseline_comparison.png"
plt.savefig(comparison_fig, dpi=300)
plt.close()

print("Saved:", comparison_fig)

# -----------------------------
# 2. Confusion matrix
# -----------------------------
confusion = pd.read_csv(v2_confusion_path, index_col=0)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(confusion.values)

ax.set_title("Improved pose-only baseline confusion matrix")
ax.set_xlabel("Predicted exercise")
ax.set_ylabel("Actual exercise")

ax.set_xticks(range(len(confusion.columns)))
ax.set_yticks(range(len(confusion.index)))
ax.set_xticklabels(confusion.columns, rotation=90)
ax.set_yticklabels(confusion.index)

for i in range(confusion.shape[0]):
    for j in range(confusion.shape[1]):
        ax.text(j, i, confusion.values[i, j], ha="center", va="center", fontsize=8)

fig.colorbar(im)
plt.tight_layout()

confusion_fig = FIGURE_DIR / "pose_baseline_v2_confusion_matrix.png"
plt.savefig(confusion_fig, dpi=300)
plt.close()

print("Saved:", confusion_fig)

# -----------------------------
# 3. Top 20 feature importances
# -----------------------------
importance = pd.read_csv(v2_importance_path)
top20 = importance.head(20).sort_values("importance")

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20["feature"], top20["importance"])
ax.set_title("Top 20 pose feature importances")
ax.set_xlabel("Random Forest importance")
ax.set_ylabel("Feature")
plt.tight_layout()

importance_fig = FIGURE_DIR / "pose_baseline_v2_feature_importance.png"
plt.savefig(importance_fig, dpi=300)
plt.close()

print("Saved:", importance_fig)

# -----------------------------
# 4. Save comparison table
# -----------------------------
comparison_out = RESULTS_DIR / "pose_baseline_comparison.csv"
comparison.to_csv(comparison_out, index=False)

print("Saved:", comparison_out)

print("\nComparison:")
print(comparison[["modality", "rows_used", "accuracy", "macro_f1"]])