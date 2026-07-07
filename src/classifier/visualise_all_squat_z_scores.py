from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = Path(r"C:\Masters\results\mmfit\squat_metrics_by_participant.csv")
OUT_DIR = Path(r"C:\Masters\results\mmfit")
FIGURE_DIR = OUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(INPUT_PATH)

# These are the exact raw squat metrics used
raw_metrics = [
    "avg_pose_movement",
    "avg_pose_range",
    "avg_pose_stability",
    "avg_watch_acc_mean",
    "avg_watch_acc_variability",
    "avg_watch_gyr_mean",
    "avg_watch_gyr_variability",
]

print("Raw squat metrics used:")
for metric in raw_metrics:
    print("-", metric)

# Create z-scores for every raw metric
z_data = data[["participant"] + raw_metrics].copy()

for metric in raw_metrics:
    mean_value = data[metric].mean()
    std_value = data[metric].std()

    z_col = metric + "_z"
    z_data[z_col] = (data[metric] - mean_value) / std_value

    print("\nMetric:", metric)
    print("Mean:", round(mean_value, 4))
    print("Std:", round(std_value, 4))

# Save full table: raw values + z-scores
output_csv = OUT_DIR / "squat_raw_metrics_and_z_scores.csv"
z_data.round(4).to_csv(output_csv, index=False)

print("\nSaved raw metrics and z-score table:")
print(output_csv)

print("\nPreview:")
print(z_data.round(4))

# ---------------------------------
# Visualise z-scores for all metrics
# ---------------------------------

z_cols = [metric + "_z" for metric in raw_metrics]

# Make one graph per metric
for metric, z_col in zip(raw_metrics, z_cols):
    plot_data = z_data.sort_values(z_col)

    plt.figure(figsize=(12, 6))
    plt.bar(plot_data["participant"], plot_data[z_col])
    plt.axhline(0, linestyle="--")
    plt.axhline(0.75, linestyle="--")
    plt.axhline(-0.75, linestyle="--")

    plt.title(f"Squat z-score by participant: {metric}")
    plt.xlabel("Participant")
    plt.ylabel("Z-score")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = FIGURE_DIR / f"squat_z_score_{metric}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Saved:", output_path)

# ---------------------------------
# Combined z-score heatmap-style table
# ---------------------------------

heatmap_data = z_data[["participant"] + z_cols].copy()
heatmap_data = heatmap_data.round(2)

fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")

table = ax.table(
    cellText=heatmap_data.values,
    colLabels=heatmap_data.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title("All squat raw metric z-scores by participant", fontsize=14, pad=20)
plt.tight_layout()

table_path = FIGURE_DIR / "squat_all_metric_z_scores_table.png"
plt.savefig(table_path, dpi=300)
plt.close()

print("\nSaved z-score table image:")
print(table_path)   