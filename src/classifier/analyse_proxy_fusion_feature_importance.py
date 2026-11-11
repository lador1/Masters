import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

from proxy_state_common import FIGURES_DIR, RESULTS_DIR, ensure_directories, save_feature_importance_figure

DATASET_PATH = RESULTS_DIR / "squat_proxy_modelling_dataset.csv"
FEATURE_GROUPS_PATH = RESULTS_DIR / "squat_proxy_feature_groups.csv"
AGG_IMPORTANCE_PATH = RESULTS_DIR / "proxy_fusion_feature_importance_aggregated.csv"
MODALITY_IMPORTANCE_PATH = RESULTS_DIR / "proxy_modality_importance.csv"
TOP_FEATURES_FIGURE_PATH = FIGURES_DIR / "proxy_fusion_top_features.png"
MODALITY_FIGURE_PATH = FIGURES_DIR / "proxy_modality_importance.png"


def build_model():
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


ensure_directories()
df = pd.read_csv(DATASET_PATH)
feature_groups = pd.read_csv(FEATURE_GROUPS_PATH)
feature_map = feature_groups[feature_groups["feature_group"].isin(["pose", "wearable", "hr"])].set_index("feature_name")["feature_group"].to_dict()
feature_cols = [c for c in df.columns if c in feature_map]
if not feature_cols:
    raise ValueError("No fused features are available for feature-importance analysis.")

target_col = "proxy_state"
all_labels = sorted(df[target_col].dropna().unique().tolist())
fold_importances = []
for train_idx, test_idx in LeaveOneGroupOut().split(df[feature_cols], df[target_col], df["participant"]):
    train_df = df.iloc[train_idx].copy()
    if not set(all_labels).issubset(set(train_df[target_col].dropna().unique())):
        continue
    model = build_model()
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train_df[target_col].astype(str)
    model.fit(X_train, y_train)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    })
    importance_df["feature_group"] = importance_df["feature"].map(feature_map)
    fold_importances.append(importance_df)

if not fold_importances:
    raise ValueError("No valid folds were available for feature-importance aggregation.")

combined = pd.concat(fold_importances, ignore_index=True)
summary = (
    combined.groupby(["feature", "feature_group"], as_index=False)["importance"]
    .agg(mean_importance="mean", std_importance="std")
    .sort_values("mean_importance", ascending=False)
    .reset_index(drop=True)
)
summary["rank"] = np.arange(1, len(summary) + 1)
summary = summary[["feature", "feature_group", "mean_importance", "std_importance", "rank"]]
summary.to_csv(AGG_IMPORTANCE_PATH, index=False)

plot_df = summary.head(20).copy().rename(columns={"mean_importance": "importance"})
save_feature_importance_figure(plot_df, TOP_FEATURES_FIGURE_PATH, title="Fusion top 20 features")

modality_summary = summary.groupby("feature_group", as_index=False)["mean_importance"].sum()
modality_summary.columns = ["feature_group", "importance_sum"]
modality_summary.to_csv(MODALITY_IMPORTANCE_PATH, index=False)

plt.figure(figsize=(7, 5))
plt.bar(modality_summary["feature_group"], modality_summary["importance_sum"], color=["#4c72b0", "#dd8452", "#55a868"])
plt.title("Proxy-state modality importance")
plt.ylabel("Importance sum")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(MODALITY_FIGURE_PATH)
plt.close()

print(summary.head(20).to_string(index=False))
print(modality_summary.to_string(index=False))
print("Saved:")
print(AGG_IMPORTANCE_PATH)
print(MODALITY_IMPORTANCE_PATH)
print(TOP_FEATURES_FIGURE_PATH)
print(MODALITY_FIGURE_PATH)
