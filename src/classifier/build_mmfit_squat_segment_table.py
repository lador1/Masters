from pathlib import Path
import pandas as pd

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

FEATURE_DIR = Path(r"C:\Masters\data\features\mmfit")
RESULTS_DIR = Path(r"C:\Masters\results\mmfit")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = FEATURE_DIR / "mmfit_fusion_features_with_hr_cleaned.csv"

OUT_PATH = FEATURE_DIR / "mmfit_squat_segment_table.csv"
SUMMARY_PATH = RESULTS_DIR / "mmfit_squat_segment_table_summary.csv"

# -------------------------------------------------------
# Load full fusion dataset
# -------------------------------------------------------

data = pd.read_csv(INPUT_PATH)

print("Loaded:")
print(INPUT_PATH)
print("Full shape:", data.shape)

# -------------------------------------------------------
# Keep squats only
# -------------------------------------------------------

squats = data[
    data["exercise"].astype(str).str.lower().str.strip() == "squats"
].copy()

if squats.empty:
    raise ValueError("No squat rows were found.")

# -------------------------------------------------------
# Remove duplicate squat segments
# -------------------------------------------------------

squats = squats.drop_duplicates(
    subset=[
        "participant",
        "start_time",
        "end_time",
        "exercise",
    ]
).copy()

# -------------------------------------------------------
# Sort each participant's squat segments in time order
# -------------------------------------------------------

squats = squats.sort_values(
    by=["participant", "start_time"]
).reset_index(drop=True)

# Squat number for each participant:
# first squat set, second squat set, third squat set
squats["squat_segment_number"] = (
    squats.groupby("participant").cumcount() + 1
)

# Meaningful ID
squats["squat_segment_id"] = (
    squats["participant"].astype(str)
    + "_squat_"
    + squats["squat_segment_number"].astype(str)
)

# -------------------------------------------------------
# Put useful identifying columns first
# -------------------------------------------------------

first_cols = [
    "squat_segment_id",
    "participant",
    "squat_segment_number",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
]

first_cols = [c for c in first_cols if c in squats.columns]

remaining_cols = [
    c for c in squats.columns
    if c not in first_cols
]

squats = squats[first_cols + remaining_cols]

# -------------------------------------------------------
# Save full squat-only table
# -------------------------------------------------------

squats.to_csv(OUT_PATH, index=False)

print("\nSaved squat segment table:")
print(OUT_PATH)

print("\nSquat table shape:")
print(squats.shape)

# -------------------------------------------------------
# Create participant summary
# -------------------------------------------------------

summary = (
    squats.groupby("participant")
    .agg(
        squat_segments=("squat_segment_id", "count"),
        total_squat_reps=("reps", "sum"),
        average_reps=("reps", "mean"),
        average_duration=("duration", "mean"),
        first_squat_start=("start_time", "min"),
        last_squat_end=("end_time", "max"),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_PATH, index=False)

print("\nSaved squat participant summary:")
print(SUMMARY_PATH)

print("\nSquat segments per participant:")
print(
    squats.groupby("participant")
    .size()
    .rename("squat_segments")
)

print("\nExercise count:")
print(squats["exercise"].value_counts())

# -------------------------------------------------------
# Simple preview
# -------------------------------------------------------

preview_cols = [
    "squat_segment_id",
    "participant",
    "squat_segment_number",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "pose_movement_mean",
    "pose_range_mean",
    "sw_l_acc_mag_mean",
    "sw_r_acc_mag_mean",
    "hr_mean_combined",
    "hr_left_right_diff",
    "hr_quality_flag",
]

preview_cols = [
    c for c in preview_cols
    if c in squats.columns
]

print("\nPreview:")
print(
    squats[preview_cols]
    .head(30)
    .to_string(index=False)
)

print("\nDone.")    