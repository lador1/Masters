import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

from proxy_state_common import (
    FIGURES_DIR,
    RESULTS_DIR,
    compute_confusion_and_report,
    ensure_directories,
    print_class_warning,
    safe_classification_metrics,
    save_confusion_figure,
    save_feature_importance_figure,
)

DATASET_PATH = RESULTS_DIR / "squat_proxy_modelling_dataset.csv"
FEATURE_GROUPS_PATH = RESULTS_DIR / "squat_proxy_feature_groups.csv"
SUMMARY_PATH = RESULTS_DIR / "proxy_fusion_summary.csv"
PREDICTIONS_PATH = RESULTS_DIR / "proxy_fusion_predictions.csv"
CONFUSION_PATH = RESULTS_DIR / "proxy_fusion_confusion_matrix.csv"
REPORT_PATH = RESULTS_DIR / "proxy_fusion_classification_report.csv"
FEATURE_IMPORTANCE_PATH = RESULTS_DIR / "proxy_fusion_feature_importance.csv"
CONFUSION_FIGURE_PATH = FIGURES_DIR / "proxy_fusion_confusion_matrix.png"
FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_DIR / "proxy_fusion_feature_importance.png"


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
    raise ValueError("No fused features are available in the modelling dataset.")

target_col = "proxy_state"
all_labels = sorted(df[target_col].dropna().unique().tolist())
if len(all_labels) < 2:
    raise ValueError("At least two proxy-state labels are required for multiclass modelling.")

pred_frames = []
final_model = None
for train_idx, test_idx in LeaveOneGroupOut().split(df[feature_cols], df[target_col], df["participant"]):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    if not set(all_labels).issubset(set(train_df[target_col].dropna().unique())):
        print(f"Skipping fold for participant(s) {sorted(test_df['participant'].unique())}: training set missing classes.")
        continue
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train_df[target_col].astype(str)
    y_test = test_df[target_col].astype(str)
    model = build_model()
    model.fit(X_train, y_train)
    final_model = model
    test_predictions = test_df[["participant", "squat_attempt_id", "squat_attempt_number"]].copy()
    test_predictions["true_proxy_state"] = y_test.values
    test_predictions["predicted_proxy_state"] = model.predict(X_test)
    pred_frames.append(test_predictions)

if not pred_frames:
    raise ValueError("No valid folds were available for participant-wise evaluation.")

pred_df = pd.concat(pred_frames, ignore_index=True)
labels = sorted(pred_df["true_proxy_state"].dropna().unique().tolist())
cm_df, report_df = compute_confusion_and_report(
    pred_df["true_proxy_state"].astype(str).tolist(),
    pred_df["predicted_proxy_state"].astype(str).tolist(),
    labels,
)
metrics = safe_classification_metrics(
    pred_df["true_proxy_state"].astype(str).to_numpy(),
    pred_df["predicted_proxy_state"].astype(str).to_numpy(),
    labels,
)
summary_df = pd.DataFrame([
    {
        "model": "Fusion",
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "n_predictions": int(len(pred_df)),
        "n_participants_tested": int(pred_df["participant"].nunique()),
    }
])
summary_df.to_csv(SUMMARY_PATH, index=False)
pred_df.to_csv(PREDICTIONS_PATH, index=False)
cm_df.to_csv(CONFUSION_PATH)
report_df = report_df.reset_index().rename(columns={"index": "proxy_state"})
report_df.to_csv(REPORT_PATH, index=False)

save_confusion_figure(cm_df, CONFUSION_FIGURE_PATH, "Fusion proxy-state confusion matrix")

feature_imp = pd.DataFrame({"feature": feature_cols, "importance": final_model.feature_importances_}).sort_values("importance", ascending=False)
feature_imp.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
save_feature_importance_figure(feature_imp, FEATURE_IMPORTANCE_FIGURE_PATH, "Fusion feature importance")

print(summary_df.to_string(index=False))
print("\nTrue class counts:")
print(pred_df["true_proxy_state"].value_counts().to_string())
print_class_warning(pred_df["true_proxy_state"])
print("Saved:")
print(SUMMARY_PATH)
print(PREDICTIONS_PATH)
print(CONFUSION_PATH)
print(REPORT_PATH)
print(FEATURE_IMPORTANCE_PATH)
print(CONFUSION_FIGURE_PATH)
print(FEATURE_IMPORTANCE_FIGURE_PATH)
