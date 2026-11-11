import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut

from proxy_state_common import FIGURES_DIR, RESULTS_DIR, ensure_directories, save_bar_chart

DATASET_PATH = RESULTS_DIR / "squat_proxy_modelling_dataset.csv"
FEATURE_GROUPS_PATH = RESULTS_DIR / "squat_proxy_feature_groups.csv"
OUTPUT_PATH = RESULTS_DIR / "proxy_ablation_results.csv"
FIGURE_PATH = FIGURES_DIR / "proxy_ablation_macro_f1.png"


def build_model():
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def evaluate_condition(feature_cols, df):
    target_col = "proxy_state"
    labels = sorted(df[target_col].dropna().unique().tolist())
    y_true = []
    y_pred = []
    for train_idx, test_idx in LeaveOneGroupOut().split(df[feature_cols], df[target_col], df["participant"]):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_classes = set(train_df[target_col].dropna().unique())
        if not set(labels).issubset(train_classes):
            print(f"Skipping fold for participant(s) {sorted(test_df['participant'].unique())}: training set missing classes.")
            continue
        model = build_model()
        model.fit(train_df[feature_cols].apply(pd.to_numeric, errors="coerce"), train_df[target_col].astype(str))
        y_true.extend(test_df[target_col].astype(str).tolist())
        y_pred.extend(model.predict(test_df[feature_cols].apply(pd.to_numeric, errors="coerce")).tolist())
    if not y_true:
        raise ValueError(f"No valid folds available for the selected condition.")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


ensure_directories()
df = pd.read_csv(DATASET_PATH)
feature_groups = pd.read_csv(FEATURE_GROUPS_PATH)
feature_map = feature_groups.set_index("feature_name")["feature_group"].to_dict()
all_features = [c for c in df.columns if c in feature_map and c not in {"participant", "squat_attempt_id", "squat_attempt_number", "proxy_state"}]
pose = [c for c in all_features if feature_map.get(c) == "pose"]
wearable = [c for c in all_features if feature_map.get(c) == "wearable"]
hr = [c for c in all_features if feature_map.get(c) == "hr"]

conditions = {
    "full_fusion": pose + wearable + hr,
    "no_pose": wearable + hr,
    "no_hr": pose + wearable,
    "no_wearable": pose + hr,
    "pose_only": pose,
    "wearable_only": wearable,
    "hr_only": hr,
}

rows = []
for name, feature_cols in conditions.items():
    if not feature_cols:
        print(f"Condition {name} has no features; skipping.")
        continue
    metrics = evaluate_condition(feature_cols, df)
    rows.append({"condition": name, **metrics})

results_df = pd.DataFrame(rows)
results_df = results_df[["condition", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]]
results_df.to_csv(OUTPUT_PATH, index=False)

save_bar_chart(results_df["macro_f1"].tolist(), results_df["condition"].tolist(), FIGURE_PATH, "Proxy-state ablation macro F1", "Condition", "Macro F1")

print(results_df.to_string(index=False))
print("Saved:")
print(OUTPUT_PATH)
print(FIGURE_PATH)
