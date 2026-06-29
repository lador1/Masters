from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(r"C:\Masters\results\mmfit")
FIGURE_DIR = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

IMPORTANCE_PATH = RESULTS_DIR / "pose_baseline_v2_fixed_feature_importance_named.csv"

importance = pd.read_csv(IMPORTANCE_PATH)

top_n = 25
top_features = importance.head(top_n).sort_values("importance", ascending=True)

plt.figure(figsize=(11, 8))
plt.barh(top_features["feature_named"], top_features["importance"])
plt.title("Top pose features for MM-Fit exercise classification")
plt.xlabel("Random Forest feature importance")
plt.ylabel("Pose feature")
plt.tight_layout()

output_path = FIGURE_DIR / "pose_baseline_v2_fixed_top_features.png"
plt.savefig(output_path, dpi=300)
plt.close()

print("Saved feature importance image to:")
print(output_path)

print("\nTop features:")
print(importance.head(top_n))