from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

INPUT_PATH = Path(r"C:\Masters\results\mmfit\squat_metrics_by_segment_usable_only.csv")
RESULTS_DIR = Path(r"C:\Masters\results\mmfit")
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = RESULTS_DIR / "squat_state_scores_by_segment.csv"
AUDIT_OUTPUT = RESULTS_DIR / "squat_state_scores_by_segment_audit.csv"
IMAGE_OUTPUT = FIGURES_DIR / "squat_state_scores_by_segment_table.png"
STATE_COUNTS_IMAGE = FIGURES_DIR / "squat_state_counts_by_segment.png"


# -------------------------------------------------------
# Load segment-level squat metrics
# -------------------------------------------------------

squats = pd.read_csv(INPUT_PATH)

print("Loaded:")
print(INPUT_PATH)
print("Shape:", squats.shape)


# -------------------------------------------------------
# Required columns
# -------------------------------------------------------

required_columns = [
    "squat_segment_id",
    "participant",
    "squat_segment_number",
    "duration_seconds",
    "reps",

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

missing_columns = [
    column for column in required_columns
    if column not in squats.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# -------------------------------------------------------
# Numeric cleanup
# -------------------------------------------------------

numeric_columns = [
    "duration_seconds",
    "reps",
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

for column in numeric_columns:
    squats[column] = pd.to_numeric(squats[column], errors="coerce")

squats[numeric_columns] = squats[numeric_columns].replace([np.inf, -np.inf], np.nan)

missing_required = squats[required_columns].isna().any(axis=1)

if missing_required.any():
    print("\nWARNING: Rows with missing required values will be removed:", int(missing_required.sum()))
    squats = squats.loc[~missing_required].copy()

if squats.empty:
    raise ValueError("No valid rows remain after removing missing required values.")


# -------------------------------------------------------
# Z-score helper
# -------------------------------------------------------

def z_score(series):
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - series.mean()) / std


# -------------------------------------------------------
# Standardise raw metrics
# -------------------------------------------------------
# z = how far this segment is above or below the group average.
# 0 = average, positive = above average, negative = below average.

squats["z_pose_movement"] = z_score(squats["pose_movement"])
squats["z_pose_range"] = z_score(squats["pose_range"])

# pose_stability is treated as instability/variation in the original pipeline.
# Higher raw value = less stable, so it contributes to variability.
squats["z_pose_instability"] = z_score(squats["pose_stability"])

squats["z_watch_acc_mean"] = z_score(squats["watch_acc_mean"])
squats["z_watch_acc_variability"] = z_score(squats["watch_acc_variability"])
squats["z_watch_gyr_mean"] = z_score(squats["watch_gyr_mean"])
squats["z_watch_gyr_variability"] = z_score(squats["watch_gyr_variability"])

squats["z_hr_mean"] = z_score(squats["hr_mean"])
squats["z_hr_change"] = z_score(squats["hr_change"])
squats["z_hr_slope"] = z_score(squats["hr_slope"])


# -------------------------------------------------------
# Original score construction method
# -------------------------------------------------------
# Keep this original method first so it matches the earlier prototype logic.

squats["movement_strength_score"] = squats[
    [
        "z_pose_movement",
        "z_pose_range",
        "z_watch_acc_mean",
        "z_watch_gyr_mean",
    ]
].mean(axis=1)

squats["movement_variability_score"] = squats[
    [
        "z_pose_instability",
        "z_watch_acc_variability",
        "z_watch_gyr_variability",
    ]
].mean(axis=1)

squats["physiological_effort_score"] = squats[
    [
        "z_hr_mean",
        "z_hr_change",
        "z_hr_slope",
    ]
].mean(axis=1)

# Original proxy logic, now applied per squat segment.
squats["engagement_like_score"] = (
    squats["movement_strength_score"]
    - squats["movement_variability_score"]
) / 2

squats["fatigue_like_score"] = (
    -squats["movement_strength_score"]
    + squats["movement_variability_score"]
    + squats["physiological_effort_score"]
) / 3

squats["frustration_like_score"] = (
    squats["movement_variability_score"]
    + squats["physiological_effort_score"]
) / 2


# -------------------------------------------------------
# Assign proxy state using original threshold order
# -------------------------------------------------------
# These are prototype thresholds:
# fatigue_like if fatigue_like_score > 0.75
# frustration_like if frustration_like_score > 0.75
# engaged_like if engagement_like_score > 0.50
# otherwise neutral_like


def assign_state(row):
    if row["fatigue_like_score"] > 0.75:
        return "fatigue_like"
    if row["frustration_like_score"] > 0.75:
        return "frustration_like"
    if row["engagement_like_score"] > 0.50:
        return "engaged_like"
    return "neutral_like"


def assign_rule(row):
    if row["fatigue_like_score"] > 0.75:
        return "fatigue_like_score > 0.75"
    if row["frustration_like_score"] > 0.75:
        return "frustration_like_score > 0.75"
    if row["engagement_like_score"] > 0.50:
        return "engagement_like_score > 0.50"
    return "no threshold exceeded"


def assign_reason(row):
    if row["proxy_state"] == "fatigue_like":
        return "Low movement strength combined with higher variability and/or physiological effort."
    if row["proxy_state"] == "frustration_like":
        return "High movement variability combined with higher physiological effort."
    if row["proxy_state"] == "engaged_like":
        return "Stronger movement with lower variability."
    return "No strong fatigue-like, frustration-like, or engaged-like signal."


squats["proxy_state"] = squats.apply(assign_state, axis=1)
squats["rule_triggered"] = squats.apply(assign_rule, axis=1)
squats["state_reason"] = squats.apply(assign_reason, axis=1)


# -------------------------------------------------------
# Output tables
# -------------------------------------------------------

output_columns = [
    "squat_segment_id",
    "participant",
    "squat_segment_number",
    "duration_seconds",
    "reps",

    "movement_strength_score",
    "movement_variability_score",
    "physiological_effort_score",

    "engagement_like_score",
    "fatigue_like_score",
    "frustration_like_score",

    "proxy_state",
    "rule_triggered",
    "state_reason",
]

z_columns = [
    "z_pose_movement",
    "z_pose_range",
    "z_pose_instability",
    "z_watch_acc_mean",
    "z_watch_acc_variability",
    "z_watch_gyr_mean",
    "z_watch_gyr_variability",
    "z_hr_mean",
    "z_hr_change",
    "z_hr_slope",
]

raw_metric_columns = [
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

score_table = squats[output_columns].copy()
audit_table = squats[
    [
        "squat_segment_id",
        "participant",
        "squat_segment_number",
        "duration_seconds",
        "reps",
    ]
    + raw_metric_columns
    + z_columns
    + [
        "movement_strength_score",
        "movement_variability_score",
        "physiological_effort_score",
        "engagement_like_score",
        "fatigue_like_score",
        "frustration_like_score",
        "proxy_state",
        "rule_triggered",
        "state_reason",
    ]
].copy()

round_cols = [
    "duration_seconds",
    "reps",
] + raw_metric_columns + z_columns + [
    "movement_strength_score",
    "movement_variability_score",
    "physiological_effort_score",
    "engagement_like_score",
    "fatigue_like_score",
    "frustration_like_score",
]

for table in [score_table, audit_table]:
    for column in round_cols:
        if column in table.columns:
            table[column] = table[column].round(4)

score_table.to_csv(CSV_OUTPUT, index=False)
audit_table.to_csv(AUDIT_OUTPUT, index=False)

print("\nSaved state score table:")
print(CSV_OUTPUT)

print("\nSaved audit table:")
print(AUDIT_OUTPUT)

print("\nState counts:")
print(score_table["proxy_state"].value_counts())

print("\nPreview:")
print(score_table.head(30).to_string(index=False))


# -------------------------------------------------------
# Visual table
# -------------------------------------------------------

visual_columns = [
    "squat_segment_id",
    "participant",
    "squat_segment_number",
    "movement_strength_score",
    "movement_variability_score",
    "physiological_effort_score",
    "engagement_like_score",
    "fatigue_like_score",
    "frustration_like_score",
    "proxy_state",
]

visual_table = score_table[visual_columns].copy()

fig_width = 22
fig_height = max(8, len(visual_table) * 0.35)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis("off")

table = ax.table(
    cellText=visual_table.values,
    colLabels=visual_table.columns,
    cellLoc="center",
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.3)

ax.set_title(
    "Squat behavioural-state proxy scores by exercise segment",
    fontsize=16,
    pad=16,
)

plt.tight_layout()
plt.savefig(IMAGE_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved visual table:")
print(IMAGE_OUTPUT)


# -------------------------------------------------------
# State count visual
# -------------------------------------------------------

state_counts = (
    score_table["proxy_state"]
    .value_counts()
    .reindex(
        [
            "engaged_like",
            "neutral_like",
            "fatigue_like",
            "frustration_like",
        ],
        fill_value=0,
    )
)

fig, ax = plt.subplots(figsize=(8, 5))
state_counts.plot(kind="bar", ax=ax)
ax.set_title("Squat proxy-state counts by segment")
ax.set_xlabel("Proxy state")
ax.set_ylabel("Segment count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(STATE_COUNTS_IMAGE, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved state-count figure:")
print(STATE_COUNTS_IMAGE)

print("\nDone.")