from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

INPUT_PATH = Path(
    r"C:\Masters\data\features\mmfit"
    r"\mmfit_fusion_features_with_hr_cleaned.csv"
)

RESULTS_DIR = Path(r"C:\Masters\results\mmfit")
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = RESULTS_DIR / "squat_metrics_by_participant_with_hr.csv"
IMAGE_OUTPUT = FIGURES_DIR / "squat_metrics_by_participant_with_hr_table.png"


# -------------------------------------------------------
# Load data and keep squats only
# -------------------------------------------------------

data = pd.read_csv(INPUT_PATH)

squats = data[
    data["exercise"].astype(str).str.strip().str.lower() == "squats"
].copy()

if squats.empty:
    raise ValueError("No squat segments were found.")


# -------------------------------------------------------
# Create the same combined watch metrics as before
# -------------------------------------------------------

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


# -------------------------------------------------------
# Remove unsuitable HR values
# -------------------------------------------------------

# Only HR marked as usable is allowed into participant averages.
usable_hr = squats["hr_quality_flag"].eq("usable")

squats["usable_hr_segment"] = usable_hr.astype(int)

hr_columns = [
    "hr_mean_combined",
    "hr_change_combined",
    "hr_slope_combined",
]

for column in hr_columns:
    if column not in squats.columns:
        raise ValueError(f"Missing required HR column: {column}")

    # Unsuitable HR becomes NaN and is ignored by participant averages.
    squats.loc[~usable_hr, column] = np.nan


# -------------------------------------------------------
# Average squat metrics by participant
# -------------------------------------------------------

participant_table = (
    squats.groupby("participant", as_index=False)
    .agg(
        squat_segments=("exercise", "size"),
        avg_reps=("reps", "mean"),
        avg_duration=("duration", "mean"),

        avg_pose_movement=("pose_movement_mean", "mean"),
        avg_pose_range=("pose_range_mean", "mean"),
        avg_pose_stability=("pose_stability_proxy", "mean"),

        avg_watch_acc_mean=("watch_acc_mean", "mean"),
        avg_watch_acc_variability=("watch_acc_variability", "mean"),
        avg_watch_gyr_mean=("watch_gyr_mean", "mean"),
        avg_watch_gyr_variability=("watch_gyr_variability", "mean"),

        usable_hr_segments=("usable_hr_segment", "sum"),
        avg_hr_mean=("hr_mean_combined", "mean"),
        avg_hr_change=("hr_change_combined", "mean"),
        avg_hr_slope=("hr_slope_combined", "mean"),
    )
)


# -------------------------------------------------------
# Round values for readability
# -------------------------------------------------------

rounding = {
    "avg_reps": 2,
    "avg_duration": 2,
    "avg_pose_movement": 4,
    "avg_pose_range": 4,
    "avg_pose_stability": 4,
    "avg_watch_acc_mean": 4,
    "avg_watch_acc_variability": 4,
    "avg_watch_gyr_mean": 4,
    "avg_watch_gyr_variability": 4,
    "avg_hr_mean": 2,
    "avg_hr_change": 2,
    "avg_hr_slope": 3,
}

participant_table = participant_table.round(rounding)


# -------------------------------------------------------
# Save the actual table as CSV
# -------------------------------------------------------

participant_table.to_csv(CSV_OUTPUT, index=False)

print("Saved participant table:")
print(CSV_OUTPUT)

print("\nShape:")
print(participant_table.shape)

print("\nUsable HR segments per participant:")
print(
    participant_table[
        ["participant", "squat_segments", "usable_hr_segments"]
    ].to_string(index=False)
)


# -------------------------------------------------------
# Create table image
# -------------------------------------------------------

display_table = participant_table.copy()

# Display missing unsuitable HR as a dash rather than "nan".
display_table = display_table.fillna("—")

fig_width = 22
fig_height = max(8, len(display_table) * 0.42)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis("off")

table = ax.table(
    cellText=display_table.values,
    colLabels=display_table.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.35)

ax.set_title(
    "Squat movement metrics by participant with usable heart-rate data",
    fontsize=16,
    pad=16,
)

plt.tight_layout()
plt.savefig(
    IMAGE_OUTPUT,
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\nSaved visual table:")
print(IMAGE_OUTPUT)

print("\nPreview:")
print(participant_table.to_string(index=False))