from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_PATH = Path(r"C:\Masters\data\features\mmfit\mmfit_fusion_features_basic.csv")
OUT_DIR = Path(r"C:\Masters\results\mmfit")
FIGURE_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(FEATURE_PATH)

# Keep only squats
squats = data[data["exercise"] == "squats"].copy()

print("Squat rows:")
print(len(squats))

# Create simple wearable summary metrics
squats["watch_acc_mean"] = squats[
    [
        "sw_l_acc_mag_mean",
        "sw_r_acc_mag_mean",
    ]
].mean(axis=1)

squats["watch_acc_variability"] = squats[
    [
        "sw_l_acc_mag_std",
        "sw_r_acc_mag_std",
    ]
].mean(axis=1)

squats["watch_gyr_mean"] = squats[
    [
        "sw_l_gyr_mag_mean",
        "sw_r_gyr_mag_mean",
    ]
].mean(axis=1)

squats["watch_gyr_variability"] = squats[
    [
        "sw_l_gyr_mag_std",
        "sw_r_gyr_mag_std",
    ]
].mean(axis=1)

# Participant-level table
summary = squats.groupby("participant").agg(
    squat_segments=("exercise", "count"),
    avg_reps=("reps", "mean"),
    avg_duration=("duration", "mean"),

    avg_pose_movement=("pose_movement_mean", "mean"),
    avg_pose_range=("pose_range_mean", "mean"),
    avg_pose_stability=("pose_stability_proxy", "mean"),

    avg_watch_acc_mean=("watch_acc_mean", "mean"),
    avg_watch_acc_variability=("watch_acc_variability", "mean"),
    avg_watch_gyr_mean=("watch_gyr_mean", "mean"),
    avg_watch_gyr_variability=("watch_gyr_variability", "mean"),
).reset_index()

# Round for easier reading
summary_rounded = summary.round(4)

# Save tables
squat_segments_path = OUT_DIR / "squat_segment_metrics.csv"
squat_summary_path = OUT_DIR / "squat_metrics_by_participant.csv"

squats.to_csv(squat_segments_path, index=False)
summary_rounded.to_csv(squat_summary_path, index=False)

print("\nSquat participant summary:")
print(summary_rounded)

print("\nSaved:")
print(squat_segments_path)
print(squat_summary_path)

# Make a visual table image
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")

table = ax.table(
    cellText=summary_rounded.values,
    colLabels=summary_rounded.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title("Squat movement metrics by participant", fontsize=14, pad=20)
plt.tight_layout()

table_image_path = FIGURE_DIR / "squat_metrics_by_participant_table.png"
plt.savefig(table_image_path, dpi=300)
plt.close()

print("\nSaved table image:")
print(table_image_path)