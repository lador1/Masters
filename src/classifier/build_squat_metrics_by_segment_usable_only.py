from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = Path(
    r"C:\Masters\data\features\mmfit"
    r"\mmfit_fusion_features_with_hr_cleaned.csv"
)

RESULTS_DIR = Path(r"C:\Masters\results\mmfit")
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = RESULTS_DIR / "squat_metrics_by_segment_usable_only.csv"
REMOVED_OUTPUT = RESULTS_DIR / "squat_segments_removed_for_quality.csv"
IMAGE_OUTPUT = FIGURES_DIR / "squat_metrics_by_segment_usable_only_table.png"

data = pd.read_csv(INPUT_PATH)

print("Loaded:")
print(INPUT_PATH)
print("Full dataset shape:", data.shape)

squats = data[
    data["exercise"].astype(str).str.strip().str.lower() == "squats"
].copy()

if squats.empty:
    raise ValueError("No squat segments were found.")

print("\nTotal squat segments before quality filtering:", len(squats))

# Preserve original squat-set order before filtering.
squats = squats.sort_values(
    by=["participant", "start_time", "end_time"]
).reset_index(drop=True)

squats["squat_segment_number"] = (
    squats.groupby("participant").cumcount() + 1
)

squats["squat_segment_id"] = (
    squats["participant"].astype(str)
    + "_squat_"
    + squats["squat_segment_number"].astype(str)
)

# MM-Fit pose/video timeline is 30 frames per second.
squats["duration_frames"] = (
    pd.to_numeric(squats["end_time"], errors="coerce")
    - pd.to_numeric(squats["start_time"], errors="coerce")
)
squats["duration_seconds"] = squats["duration_frames"] / 30.0

required_watch_columns = [
    "sw_l_acc_mag_mean",
    "sw_r_acc_mag_mean",
    "sw_l_acc_mag_std",
    "sw_r_acc_mag_std",
    "sw_l_gyr_mag_mean",
    "sw_r_gyr_mag_mean",
    "sw_l_gyr_mag_std",
    "sw_r_gyr_mag_std",
]

missing_watch_columns = [
    column for column in required_watch_columns
    if column not in squats.columns
]

if missing_watch_columns:
    raise ValueError(f"Missing required watch columns: {missing_watch_columns}")

squats["watch_acc_mean"] = squats[
    ["sw_l_acc_mag_mean", "sw_r_acc_mag_mean"]
].mean(axis=1)

squats["watch_acc_variability"] = squats[
    ["sw_l_acc_mag_std", "sw_r_acc_mag_std"]
].mean(axis=1)

squats["watch_gyr_mean"] = squats[
    ["sw_l_gyr_mag_mean", "sw_r_gyr_mag_mean"]
].mean(axis=1)

squats["watch_gyr_variability"] = squats[
    ["sw_l_gyr_mag_std", "sw_r_gyr_mag_std"]
].mean(axis=1)

required_metric_columns = [
    "duration_frames",
    "duration_seconds",
    "pose_movement_mean",
    "pose_range_mean",
    "pose_stability_proxy",
    "watch_acc_mean",
    "watch_acc_variability",
    "watch_gyr_mean",
    "watch_gyr_variability",
    "hr_mean_combined",
    "hr_change_combined",
    "hr_slope_combined",
]

missing_required_columns = [
    column for column in required_metric_columns
    if column not in squats.columns
]

if missing_required_columns:
    raise ValueError(f"Missing required metric columns: {missing_required_columns}")

if "hr_quality_flag" not in squats.columns:
    raise ValueError("Missing hr_quality_flag column.")

squats["removal_reason"] = ""

bad_hr_quality = (
    squats["hr_quality_flag"]
    .astype(str)
    .str.strip()
    .str.lower()
    .ne("usable")
)
squats.loc[bad_hr_quality, "removal_reason"] = "unusable_or_missing_hr"

for column in required_metric_columns:
    squats[column] = pd.to_numeric(squats[column], errors="coerce")

squats[required_metric_columns] = squats[
    required_metric_columns
].replace([np.inf, -np.inf], np.nan)

missing_required_metric = squats[
    required_metric_columns
].isna().any(axis=1)

squats.loc[
    missing_required_metric & squats["removal_reason"].eq(""),
    "removal_reason"
] = "missing_required_metric"

squats.loc[
    missing_required_metric
    & squats["removal_reason"].ne("")
    & ~squats["removal_reason"].str.contains("missing_required_metric", na=False),
    "removal_reason"
] += ";missing_required_metric"

removed = squats[squats["removal_reason"].ne("")].copy()

removed_columns = [
    "squat_segment_id",
    "participant",
    "squat_segment_number",
    "start_time",
    "end_time",
    "duration_frames",
    "duration_seconds",
    "reps",
    "exercise",
    "hr_quality_flag",
    "hr_left_right_diff",
    "removal_reason",
]
removed_columns = [column for column in removed_columns if column in removed.columns]

removed[removed_columns].to_csv(REMOVED_OUTPUT, index=False)

print("\nRemoved squat segments:", len(removed))
print("Saved removed-segment audit file:")
print(REMOVED_OUTPUT)

if not removed.empty:
    print("\nRemoval reasons:")
    print(removed["removal_reason"].value_counts())

usable = squats[squats["removal_reason"].eq("")].copy()

if usable.empty:
    raise ValueError("No squat segments remain after quality filtering.")

usable = usable.sort_values(
    by=["participant", "squat_segment_number"]
).reset_index(drop=True)

segment_table = usable[
    [
        "squat_segment_id",
        "participant",
        "squat_segment_number",
        "start_time",
        "end_time",
        "duration_frames",
        "duration_seconds",
        "reps",
        "pose_movement_mean",
        "pose_range_mean",
        "pose_stability_proxy",
        "watch_acc_mean",
        "watch_acc_variability",
        "watch_gyr_mean",
        "watch_gyr_variability",
        "hr_mean_combined",
        "hr_change_combined",
        "hr_slope_combined",
        "hr_quality_flag",
    ]
].copy()

segment_table = segment_table.rename(
    columns={
        "pose_movement_mean": "pose_movement",
        "pose_range_mean": "pose_range",
        "pose_stability_proxy": "pose_stability",
        "hr_mean_combined": "hr_mean",
        "hr_change_combined": "hr_change",
        "hr_slope_combined": "hr_slope",
    }
)

segment_table = segment_table.round(
    {
        "duration_seconds": 2,
        "reps": 2,
        "pose_movement": 4,
        "pose_range": 4,
        "pose_stability": 4,
        "watch_acc_mean": 4,
        "watch_acc_variability": 4,
        "watch_gyr_mean": 4,
        "watch_gyr_variability": 4,
        "hr_mean": 2,
        "hr_change": 2,
        "hr_slope": 3,
    }
)

segment_table.to_csv(CSV_OUTPUT, index=False)

print("\nSaved segment-level squat metrics:")
print(CSV_OUTPUT)
print("\nFinal table shape:", segment_table.shape)
print("\nParticipants remaining:", segment_table["participant"].nunique())
print("\nUsable squat segments remaining:", len(segment_table))

print("\nUsable segments per participant:")
print(
    segment_table.groupby("participant")
    .size()
    .rename("usable_squat_segments")
    .to_string()
)

display_columns = [
    "squat_segment_id",
    "participant",
    "squat_segment_number",
    "reps",
    "duration_seconds",
    "pose_movement",
    "pose_range",
    "pose_stability",
    "watch_acc_mean",
    "watch_acc_variability",
    "watch_gyr_mean",
    "watch_gyr_variability",
    "hr_mean",
    "hr_change",
    "hr_slope",
]

display_table = segment_table[display_columns].copy()

fig_width = 24
fig_height = max(8, len(display_table) * 0.35)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis("off")

table = ax.table(
    cellText=display_table.values,
    colLabels=display_table.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.3)

ax.set_title(
    "Squat metrics by individual exercise segment — usable data only",
    fontsize=16,
    pad=16,
)

plt.tight_layout()
plt.savefig(IMAGE_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved visual table:")
print(IMAGE_OUTPUT)

print("\nPreview:")
print(segment_table.head(30).to_string(index=False))
print("\nDone.")