import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from proxy_state_common import FIGURES_DIR, RESULTS_DIR, ensure_directories

MODEL_FILES = {
    "Pose only": RESULTS_DIR / "proxy_pose_only_summary.csv",
    "Wearable + HR": RESULTS_DIR / "proxy_wearable_hr_summary.csv",
    "Fusion": RESULTS_DIR / "proxy_fusion_summary.csv",
}
COMPARISON_PATH = RESULTS_DIR / "proxy_model_comparison.csv"
FIG_ACC_PATH = FIGURES_DIR / "proxy_model_accuracy_comparison.png"
FIG_MACRO_PATH = FIGURES_DIR / "proxy_model_macro_f1_comparison.png"
FIG_BAL_PATH = FIGURES_DIR / "proxy_model_balanced_accuracy_comparison.png"


ensure_directories()
rows = []
for model_name, path in MODEL_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file for {model_name}: {path}")
    summary = pd.read_csv(path)
    if summary.empty:
        raise ValueError(f"Summary file is empty for {model_name}: {path}")
    row = summary.iloc[0].copy()
    row["model"] = model_name
    rows.append(row)

comparison_df = pd.DataFrame(rows)[["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "n_predictions", "n_participants_tested"]].copy()
comparison_df.to_csv(COMPARISON_PATH, index=False)

print(comparison_df.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.bar(comparison_df["model"], comparison_df["accuracy"], color=["#4c72b0", "#dd8452", "#55a868"])
plt.title("Proxy-state accuracy by model")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIG_ACC_PATH)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(comparison_df["model"], comparison_df["macro_f1"], color=["#4c72b0", "#dd8452", "#55a868"])
plt.title("Proxy-state macro F1 by model")
plt.ylabel("Macro F1")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIG_MACRO_PATH)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(comparison_df["model"], comparison_df["balanced_accuracy"], color=["#4c72b0", "#dd8452", "#55a868"])
plt.title("Proxy-state balanced accuracy by model")
plt.ylabel("Balanced accuracy")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(FIG_BAL_PATH)
plt.close()

best_row = comparison_df.sort_values("macro_f1", ascending=False).iloc[0]
if best_row["model"] == "Fusion":
    conclusion = "Fusion produced the strongest proxy-state classification performance."
elif best_row["model"] == "Wearable + HR":
    conclusion = "Wearable + HR produced the strongest classification performance; fusion did not improve over the wearable condition."
else:
    conclusion = "Pose-only produced the strongest classification performance."
print(conclusion)
print("Saved:")
print(COMPARISON_PATH)
print(FIG_ACC_PATH)
print(FIG_MACRO_PATH)
print(FIG_BAL_PATH)
