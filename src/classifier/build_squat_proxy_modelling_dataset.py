from pathlib import Path

import pandas as pd

from proxy_state_common import DATA_DIR, RESULTS_DIR, detect_join_key, ensure_directories, select_feature_columns

FEATURE_PATH = DATA_DIR / "mmfit_fusion_features_with_hr_cleaned.csv"
SEGMENT_PATH = RESULTS_DIR / "squat_metrics_by_segment_usable_only.csv"
LABEL_PATH = RESULTS_DIR / "squat_state_scores_per_participant_exercise_attempt.csv"
MODELLING_PATH = RESULTS_DIR / "squat_proxy_modelling_dataset.csv"
FEATURE_GROUPS_PATH = RESULTS_DIR / "squat_proxy_feature_groups.csv"


def normalise_segment_labels(segment_df, label_df):
    seg = segment_df.copy()
    rename_map = {}
    if "squat_segment_id" in seg.columns:
        rename_map["squat_segment_id"] = "squat_attempt_id"
    if "squat_segment_number" in seg.columns:
        rename_map["squat_segment_number"] = "squat_attempt_number"
    seg = seg.rename(columns=rename_map)
    label_cols = [
        c for c in ["participant", "squat_attempt_id", "squat_attempt_number", "proxy_state"]
        if c in label_df.columns
    ]
    if "proxy_state" not in label_cols:
        raise ValueError("proxy_state is missing from the label dataframe.")
    merge_on = [c for c in ["participant", "squat_attempt_id", "squat_attempt_number"] if c in label_cols]
    return seg.merge(label_df[label_cols], on=merge_on, how="inner")


ensure_directories()
feature_df = pd.read_csv(FEATURE_PATH)
segment_df = pd.read_csv(SEGMENT_PATH)
label_df = pd.read_csv(LABEL_PATH)

required_feature_cols = ["participant", "start_time", "end_time"]
required_label_cols = ["participant", "proxy_state"]
missing_feature_cols = [c for c in required_feature_cols if c not in feature_df.columns]
missing_label_cols = [c for c in required_label_cols if c not in label_df.columns]
if missing_feature_cols or missing_label_cols:
    raise ValueError(f"Missing required columns. Feature missing: {missing_feature_cols}; Label missing: {missing_label_cols}")

print("Loaded feature file:", FEATURE_PATH)
print("Feature shape:", feature_df.shape)
print("Segment shape:", segment_df.shape)
print("Label shape:", label_df.shape)
print("Using join keys for feature -> segment table:", detect_join_key(feature_df, segment_df))

segment_with_labels = normalise_segment_labels(segment_df, label_df)
print("Segment + label shape:", segment_with_labels.shape)

merge_cols = [c for c in ["participant", "start_time", "end_time"] if c in feature_df.columns and c in segment_with_labels.columns]
if len(merge_cols) < 2:
    raise ValueError("Could not find a valid feature-to-segment join key. Need at least participant plus one time column.")

joined = feature_df.merge(
    segment_with_labels[["participant", "start_time", "end_time", "squat_attempt_id", "squat_attempt_number", "proxy_state"]],
    on=merge_cols,
    how="inner",
)
print("Matched rows shape:", joined.shape)

metadata_exclusions = {
    "participant",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
    "pose_valid",
    "pose_frames",
    "pose_joint_count",
    "hr_left_available",
    "hr_right_available",
    "hr_any_available",
    "hr_both_available",
    "hr_left_right_diff",
    "hr_disagreement_flag",
    "hr_quality_flag",
    "squat_attempt_id",
    "squat_attempt_number",
    "proxy_state",
}

raw_feature_cols = [
    col for col in joined.columns
    if col not in metadata_exclusions
    and not col.lower().endswith("_score")
    and "state" not in col.lower()
    and "threshold" not in col.lower()
    and "rule" not in col.lower()
    and "label" not in col.lower()
    and "timestamp" not in col.lower()
    and "duration" not in col.lower()
    and "frame" not in col.lower()
]

if not raw_feature_cols:
    raise ValueError("No usable raw feature columns were found.")

modelling_df = joined[["participant", "squat_attempt_id", "squat_attempt_number", "proxy_state"] + raw_feature_cols].copy()
modelling_df = modelling_df.dropna(subset=["proxy_state"]).copy()

feature_groups = select_feature_columns(modelling_df)
feature_group_rows = [
    {"feature_name": feature_name, "feature_group": feature_groups.get(feature_name, "unknown")}
    for feature_name in raw_feature_cols
    if feature_name in feature_groups
]
feature_group_df = pd.DataFrame(feature_group_rows)
feature_group_df = feature_group_df[feature_group_df["feature_group"].isin(["pose", "wearable", "hr"])].copy()
feature_group_df = feature_group_df.drop_duplicates().sort_values(["feature_group", "feature_name"]).reset_index(drop=True)

modelling_df.to_csv(MODELLING_PATH, index=False)
feature_group_df.to_csv(FEATURE_GROUPS_PATH, index=False)

pose_count = int((feature_group_df["feature_group"] == "pose").sum())
wearable_count = int((feature_group_df["feature_group"] == "wearable").sum())
hr_count = int((feature_group_df["feature_group"] == "hr").sum())

print("\nFinal dataset summary:")
print("shape:", modelling_df.shape)
print("participants:", modelling_df["participant"].nunique())
print("class counts:\n", modelling_df["proxy_state"].value_counts().to_string())
print("pose features:", pose_count)
print("wearable features:", wearable_count)
print("HR features:", hr_count)
print("\nSaved:")
print(MODELLING_PATH)
print(FEATURE_GROUPS_PATH)
