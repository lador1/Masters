from pathlib import Path
import pandas as pd

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

FEATURE_DIR = Path(r"C:\Masters\data\features\mmfit")
RESULTS_DIR = Path(r"C:\Masters\results\mmfit")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = FEATURE_DIR / "mmfit_fusion_features_with_hr_cleaned.csv"
OUT_PATH = FEATURE_DIR / "mmfit_master_segment_table.csv"
SUMMARY_PATH = RESULTS_DIR / "mmfit_master_segment_table_summary.csv"
COLUMN_GROUPS_PATH = RESULTS_DIR / "mmfit_master_segment_table_column_groups.csv"

# -------------------------------------------------------
# Load cleaned fusion + HR file
# -------------------------------------------------------

data = pd.read_csv(INPUT_PATH)

print("Loaded:")
print(INPUT_PATH)
print("Shape:", data.shape)

# -------------------------------------------------------
# Basic safety checks
# -------------------------------------------------------

required_cols = [
    "participant",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
]

missing_required = [c for c in required_cols if c not in data.columns]

if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# -------------------------------------------------------
# Remove duplicate rows if any
# -------------------------------------------------------

before = len(data)

data = data.drop_duplicates(
    subset=[
        "participant",
        "start_time",
        "end_time",
        "exercise",
    ]
).copy()

after = len(data)

print("\nDuplicate check:")
print("Rows before:", before)
print("Rows after:", after)
print("Duplicates removed:", before - after)

# -------------------------------------------------------
# Create useful IDs
# -------------------------------------------------------

data = data.sort_values(
    by=["participant", "start_time", "end_time"]
).reset_index(drop=True)

data.insert(
    0,
    "segment_id",
    ["seg_" + str(i).zfill(4) for i in range(len(data))]
)

data["participant_segment_number"] = (
    data.groupby("participant").cumcount() + 1
)

# -------------------------------------------------------
# Organise columns into cleaner order
# -------------------------------------------------------

metadata_cols = [
    "segment_id",
    "participant",
    "participant_segment_number",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
]

hr_quality_cols = [
    "hr_left_available",
    "hr_right_available",
    "hr_any_available",
    "hr_both_available",
    "hr_left_right_diff",
    "hr_disagreement_flag",
    "hr_quality_flag",
]

hr_combined_cols = [
    "hr_mean_combined",
    "hr_std_combined",
    "hr_min_combined",
    "hr_max_combined",
    "hr_range_combined",
    "hr_change_combined",
    "hr_slope_combined",
]

pose_cols = [
    c for c in data.columns
    if c.startswith("pose_") or c.startswith("joint_")
]

wearable_cols = [
    c for c in data.columns
    if (
        c.startswith("sw_l_acc_")
        or c.startswith("sw_r_acc_")
        or c.startswith("sw_l_gyr_")
        or c.startswith("sw_r_gyr_")
    )
]

raw_hr_cols = [
    c for c in data.columns
    if c.startswith("sw_l_hr_") or c.startswith("sw_r_hr_")
]

known_cols = (
    metadata_cols
    + pose_cols
    + wearable_cols
    + raw_hr_cols
    + hr_combined_cols
    + hr_quality_cols
)

known_cols = [c for c in known_cols if c in data.columns]

other_cols = [c for c in data.columns if c not in known_cols]

ordered_cols = known_cols + other_cols

master = data[ordered_cols].copy()

# -------------------------------------------------------
# Save master table
# -------------------------------------------------------

master.to_csv(OUT_PATH, index=False)

print("\nSaved master table:")
print(OUT_PATH)
print("Shape:", master.shape)

# -------------------------------------------------------
# Create summary file
# -------------------------------------------------------

summary_rows = []

summary_rows.append({
    "metric": "total_rows",
    "value": len(master),
    "description": "Total exercise segments in the master table"
})

summary_rows.append({
    "metric": "total_columns",
    "value": len(master.columns),
    "description": "Total feature and metadata columns"
})

summary_rows.append({
    "metric": "participants",
    "value": master["participant"].nunique(),
    "description": "Number of participants"
})

summary_rows.append({
    "metric": "exercise_classes",
    "value": master["exercise"].nunique(),
    "description": "Number of exercise types"
})

summary_rows.append({
    "metric": "pose_columns",
    "value": len(pose_cols),
    "description": "Pose-derived movement-quality feature columns"
})

summary_rows.append({
    "metric": "wearable_columns",
    "value": len(wearable_cols),
    "description": "Smartwatch accelerometer and gyroscope feature columns"
})

summary_rows.append({
    "metric": "raw_hr_columns",
    "value": len(raw_hr_cols),
    "description": "Original left/right heart-rate feature columns"
})

summary_rows.append({
    "metric": "combined_hr_columns",
    "value": len(hr_combined_cols),
    "description": "Cleaned combined heart-rate feature columns"
})

if "hr_any_available" in master.columns:
    summary_rows.append({
        "metric": "segments_with_any_hr",
        "value": int(master["hr_any_available"].sum()),
        "description": "Segments where at least one HR signal is available"
    })

if "hr_disagreement_flag" in master.columns:
    summary_rows.append({
        "metric": "segments_with_hr_disagreement",
        "value": int(master["hr_disagreement_flag"].sum()),
        "description": "Segments where left/right HR differ by more than 20 bpm"
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_PATH, index=False)

print("\nSaved summary:")
print(SUMMARY_PATH)
print(summary)

# -------------------------------------------------------
# Save column group map
# -------------------------------------------------------

column_groups = []

for col in metadata_cols:
    if col in master.columns:
        column_groups.append({
            "column": col,
            "group": "metadata",
            "description": "Segment identity, participant, timing, reps, or exercise label"
        })

for col in pose_cols:
    column_groups.append({
        "column": col,
        "group": "pose",
        "description": "Pose-derived movement-quality feature"
    })

for col in wearable_cols:
    column_groups.append({
        "column": col,
        "group": "wearable_acc_gyr",
        "description": "Smartwatch accelerometer or gyroscope feature"
    })

for col in raw_hr_cols:
    column_groups.append({
        "column": col,
        "group": "raw_heart_rate",
        "description": "Original left/right heart-rate feature"
    })

for col in hr_combined_cols:
    if col in master.columns:
        column_groups.append({
            "column": col,
            "group": "combined_heart_rate",
            "description": "Cleaned combined heart-rate feature"
        })

for col in hr_quality_cols:
    if col in master.columns:
        column_groups.append({
            "column": col,
            "group": "heart_rate_quality",
            "description": "Heart-rate availability or reliability flag"
        })

for col in other_cols:
    column_groups.append({
        "column": col,
        "group": "other",
        "description": "Other feature retained from previous processing"
    })

column_groups_df = pd.DataFrame(column_groups)
column_groups_df.to_csv(COLUMN_GROUPS_PATH, index=False)

print("\nSaved column groups:")
print(COLUMN_GROUPS_PATH)

# -------------------------------------------------------
# Print useful checks
# -------------------------------------------------------

print("\nExercise counts:")
print(master["exercise"].value_counts())

print("\nParticipant counts:")
print(master["participant"].value_counts().sort_index())

if "hr_quality_flag" in master.columns:
    print("\nHR quality counts:")
    print(master["hr_quality_flag"].value_counts())

print("\nPreview:")
preview_cols = [
    "segment_id",
    "participant",
    "participant_segment_number",
    "exercise",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "hr_mean_combined",
    "hr_left_right_diff",
    "hr_quality_flag",
]

preview_cols = [c for c in preview_cols if c in master.columns]
print(master[preview_cols].head(30).to_string(index=False))

print("\nDone.")