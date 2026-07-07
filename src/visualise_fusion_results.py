from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(r"C:\Masters\results\mmfit")
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# 1. Model comparison data
# -------------------------------------------------------

comparison = pd.DataFrame([
    {
        "model": "Pose-only",
        "rows": 579,
        "features": 82,
        "accuracy": 0.44,
        "macro_f1": 0.45,
    },
    {
        "model": "Wearable-only",
        "rows": 616,
        "features": 40,
        "accuracy": 0.9758,
        "macro_f1": 0.9769,
    },
    {
        "model": "Fusion",
        "rows": 579,
        "features": 118,
        "accuracy": 0.9741,
        "macro_f1": 0.9750,
    },
])

comparison_path = RESULTS_DIR / "model_comparison_pose_wearable_fusion.csv"
comparison.to_csv(comparison_path, index=False)

print("Saved model comparison table:")
print(comparison_path)

# -------------------------------------------------------
# 2. Accuracy comparison graph
# -------------------------------------------------------

plt.figure(figsize=(9, 6))
plt.bar(comparison["model"], comparison["accuracy"])
plt.title("Model accuracy comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)

for i, value in enumerate(comparison["accuracy"]):
    plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

plt.tight_layout()

accuracy_path = FIGURE_DIR / "model_accuracy_comparison.png"
plt.savefig(accuracy_path, dpi=300)
plt.close()

print("Saved:")
print(accuracy_path)

# -------------------------------------------------------
# 3. Macro F1 comparison graph
# -------------------------------------------------------

plt.figure(figsize=(9, 6))
plt.bar(comparison["model"], comparison["macro_f1"])
plt.title("Model macro F1 comparison")
plt.xlabel("Model")
plt.ylabel("Macro F1-score")
plt.ylim(0, 1.05)

for i, value in enumerate(comparison["macro_f1"]):
    plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

plt.tight_layout()

f1_path = FIGURE_DIR / "model_macro_f1_comparison.png"
plt.savefig(f1_path, dpi=300)
plt.close()

print("Saved:")
print(f1_path)

# -------------------------------------------------------
# 4. Fusion feature importance graph
# -------------------------------------------------------

importance_path = RESULTS_DIR / "fusion_baseline_basic_feature_importance.csv"

importance = pd.read_csv(importance_path)

top_n = 25
top_features = importance.head(top_n).sort_values("importance", ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(top_features["feature"], top_features["importance"])
plt.title("Top fusion model features")
plt.xlabel("Random Forest feature importance")
plt.ylabel("Feature")
plt.tight_layout()

fusion_importance_path = FIGURE_DIR / "fusion_top_feature_importance.png"
plt.savefig(fusion_importance_path, dpi=300)
plt.close()

print("Saved:")
print(fusion_importance_path)

print("\nTop fusion features:")
print(importance.head(top_n))