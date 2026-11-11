from pathlib import Path

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

CSV_OUTPUT = RESULTS_DIR / "squat_metrics_by_participant_usable_hr_only.csv"
IMAGE_OUTPUT = (
    FIGURES_DIR
    / "squat_metrics_by_participant_usable_hr_only_table.png"
)


# -------------------------------------------------------
# Load data
# -------------------------------------------------------

data = pd.read_csv(INPUT_PATH)

# Keep squats only
squats = data[
    data["exercise"].astype(str).str.strip().str.lower() == "squats"
].copy()

if squats.empty:
    raise ValueError("No squat segments were found.")


# -------------------------------------------------------
# Remove unusable HR segments completely
# -------------------------------------------------------

before_count = len(squats)

squats = squats[
    squats["hr_quality_flag"].astype(str).str.strip().str.lower() == "usable"
].copy()

after_count = len(squats)

print("Squat segments before HR filtering:", before_count)
print("Squat segments after HR filtering:", after_count)
print("Removed unusable HR segments:", before_count - after_count)

if squats.empty:
    raise ValueError("No squat segments remain after HR filtering.")


# -------------------------------------------------------
# Create combined smartwatch movement metrics
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
# Average remaining squat segments by participant
# -------------------------------------------------------

participant_table = (
    squats.groupby("participant", as_index=False)
    .agg(
        usable_squat_segments=("exercise", "size"),
        avg_reps=("reps", "mean"),
        avg_duration=("duration", "mean"),

        avg_pose_movement=("pose_movement_mean", "mean"),
        avg_pose_range=("pose_range_mean", "mean"),
        avg_pose_stability=("pose_stability_proxy", "mean"),

        avg_watch_acc_mean=("watch_acc_mean", "mean"),
        avg_watch_acc_variability=("watch_acc_variability", "mean"),
        avg_watch_gyr_mean=("watch_gyr_mean", "mean"),
        avg_watch_gyr_variability=("watch_gyr_variability", "mean"),

        avg_hr_mean=("hr_mean_combined", "mean"),
        avg_hr_change=("hr_change_combined", "mean"),
        avg_hr_slope=("hr_slope_combined", "mean"),
    )
)


# -------------------------------------------------------
# Round values
# -------------------------------------------------------

participant_table = participant_table.round({
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
})


# -------------------------------------------------------
# Save CSV
# -------------------------------------------------------

participant_table.to_csv(CSV_OUTPUT, index=False)

print("\nSaved CSV:")
print(CSV_OUTPUT)

print("\nParticipants remaining:")
print(participant_table["participant"].nunique())

print("\nParticipant table:")
print(participant_table.to_string(index=False))


# -------------------------------------------------------
# Create visual table
# -------------------------------------------------------

fig_width = 22
fig_height = max(7, len(participant_table) * 0.45)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis("off")

table = ax.table(
    cellText=participant_table.values,
    colLabels=participant_table.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.35)

ax.set_title(
    "Squat movement metrics by participant — usable HR segments only",
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

print("\nDone.")