from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

INPUT_PATH = Path(
    r"C:\Masters\results\mmfit"
    r"\squat_metrics_by_segment_usable_only.csv"
)

RESULTS_DIR = Path(r"C:\Masters\results\mmfit")
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = RESULTS_DIR / "squat_state_scores_per_participant_exercise_attempt.csv"
AUDIT_OUTPUT = RESULTS_DIR / "squat_state_scores_per_participant_exercise_attempt_audit.csv"
IMAGE_OUTPUT = FIGURES_DIR / "squat_state_scores_per_participant_exercise_attempt_table.png"
COUNT_FIGURE_OUTPUT = FIGURES_DIR / "squat_state_counts_per_participant_exercise_attempt.png"


# -------------------------------------------------------
# Load squat attempt metrics
# -------------------------------------------------------

attempts = pd.read_csv(INPUT_PATH)

print("Loaded:")
print(INPUT_PATH)
print("Shape:", attempts.shape)

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
    if column not in attempts.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# -------------------------------------------------------
# Rename segment wording to exercise-attempt wording
# -------------------------------------------------------

attempts = attempts.rename(
    columns={
        "squat_segment_id": "squat_attempt_id",
        "squat_segment_number": "squat_attempt_number",
    }
)


# -------------------------------------------------------
# Convert scoring columns to numeric
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
    attempts[column] = pd.to_numeric(attempts[column], errors="coerce")

attempts[numeric_columns] = attempts[numeric_columns].replace(
    [np.inf, -np.inf],
    np.nan,
)

missing_metric_rows = attempts[numeric_columns].isna().any(axis=1)

if missing_metric_rows.any():
    print("\nWarning: rows with missing metrics were removed before scoring:")
    print(attempts.loc[missing_metric_rows, ["squat_attempt_id", "participant"]])
    attempts = attempts.loc[~missing_metric_rows].copy()

if attempts.empty:
    raise ValueError("No rows remain after removing missing metrics.")


# -------------------------------------------------------
# Z-score helper
# -------------------------------------------------------

def z_score(series):
    std = series.std(ddof=0)

    if std == 0 or np.isnan(std):
        return series * 0

    return (series - series.mean()) / std


# -------------------------------------------------------
# Standardised metric scores
# -------------------------------------------------------

# Movement strength indicators
attempts["z_pose_movement"] = z_score(attempts["pose_movement"])
attempts["z_pose_range"] = z_score(attempts["pose_range"])
attempts["z_watch_acc_mean"] = z_score(attempts["watch_acc_mean"])
attempts["z_watch_gyr_mean"] = z_score(attempts["watch_gyr_mean"])

# Movement variability / instability indicators
attempts["z_pose_instability"] = z_score(attempts["pose_stability"])
attempts["z_watch_acc_variability"] = z_score(attempts["watch_acc_variability"])
attempts["z_watch_gyr_variability"] = z_score(attempts["watch_gyr_variability"])

# Physiological effort indicators from cleaned heart rate
attempts["z_hr_mean"] = z_score(attempts["hr_mean"])
attempts["z_hr_change"] = z_score(attempts["hr_change"])
attempts["z_hr_slope"] = z_score(attempts["hr_slope"])


# -------------------------------------------------------
# Original scoring structure, now per squat exercise attempt
# -------------------------------------------------------

attempts["movement_strength_score"] = attempts[
    [
        "z_pose_movement",
        "z_pose_range",
        "z_watch_acc_mean",
        "z_watch_gyr_mean",
    ]
].mean(axis=1)

attempts["movement_variability_score"] = attempts[
    [
        "z_pose_instability",
        "z_watch_acc_variability",
        "z_watch_gyr_variability",
    ]
].mean(axis=1)

attempts["physiological_effort_score"] = attempts[
    [
        "z_hr_mean",
        "z_hr_change",
        "z_hr_slope",
    ]
].mean(axis=1)

attempts["engagement_like_score"] = (
    attempts["movement_strength_score"]
    - attempts["movement_variability_score"]
) / 2

attempts["fatigue_like_score"] = (
    -attempts["movement_strength_score"]
    + attempts["movement_variability_score"]
    + attempts["physiological_effort_score"]
) / 3

attempts["frustration_like_score"] = (
    attempts["movement_variability_score"]
    + attempts["physiological_effort_score"]
) / 2


# -------------------------------------------------------
# Proxy-state rules
# These are conservative rule-based proxy labels, not true emotion labels.
# -------------------------------------------------------

def assign_proxy_state(row):
    if row["fatigue_like_score"] > 0.75:
        return "fatigue_like"
    if row["frustration_like_score"] > 0.75:
        return "frustration_like"
    if row["engagement_like_score"] > 0.50:
        return "engaged_like"
    return "neutral_like"


def rule_triggered(row):
    if row["fatigue_like_score"] > 0.75:
        return "fatigue_like_score > 0.75"
    if row["frustration_like_score"] > 0.75:
        return "frustration_like_score > 0.75"
    if row["engagement_like_score"] > 0.50:
        return "engagement_like_score > 0.50"
    return "no threshold exceeded"


def state_reason(row):
    state = row["proxy_state"]

    if state == "fatigue_like":
        return "Lower movement strength combined with higher variability and physiological effort."
    if state == "frustration_like":
        return "High movement variability combined with higher physiological effort."
    if state == "engaged_like":
        return "Stronger movement pattern with lower variability."
    return "No strong fatigue-like, frustration-like, or engaged-like signal."


attempts["proxy_state"] = attempts.apply(assign_proxy_state, axis=1)
attempts["rule_triggered"] = attempts.apply(rule_triggered, axis=1)
attempts["state_reason"] = attempts.apply(state_reason, axis=1)


# -------------------------------------------------------
# Final output table
# -------------------------------------------------------

output_columns = [
    "squat_attempt_id",
    "participant",
    "squat_attempt_number",
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

score_table = attempts[output_columns].copy()

score_table = score_table.round(
    {
        "duration_seconds": 2,
        "reps": 2,
        "movement_strength_score": 4,
        "movement_variability_score": 4,
        "physiological_effort_score": 4,
        "engagement_like_score": 4,
        "fatigue_like_score": 4,
        "frustration_like_score": 4,
    }
)

score_table.to_csv(CSV_OUTPUT, index=False)

print("\nSaved state scores:")
print(CSV_OUTPUT)
print("Shape:", score_table.shape)


# -------------------------------------------------------
# Audit output with raw metrics and z-scores
# -------------------------------------------------------

audit_columns = [
    "squat_attempt_id",
    "participant",
    "squat_attempt_number",
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

    "z_pose_movement",
    "z_pose_range",
    "z_watch_acc_mean",
    "z_watch_gyr_mean",
    "z_pose_instability",
    "z_watch_acc_variability",
    "z_watch_gyr_variability",
    "z_hr_mean",
    "z_hr_change",
    "z_hr_slope",

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

audit = attempts[audit_columns].copy()
audit = audit.round(4)
audit.to_csv(AUDIT_OUTPUT, index=False)

print("\nSaved audit table:")
print(AUDIT_OUTPUT)


# -------------------------------------------------------
# Visual table
# -------------------------------------------------------

visual_columns = [
    "squat_attempt_id",
    "participant",
    "squat_attempt_number",
    "duration_seconds",
    "movement_strength_score",
    "movement_variability_score",
    "physiological_effort_score",
    "engagement_like_score",
    "fatigue_like_score",
    "frustration_like_score",
    "proxy_state",
]

visual_table = score_table[visual_columns].copy()

fig_width = 24
fig_height = max(8, len(visual_table) * 0.34)

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
    "Squat state scores per participant exercise attempt",
    fontsize=16,
    pad=16,
)

plt.tight_layout()
plt.savefig(IMAGE_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved visual table:")
print(IMAGE_OUTPUT)


# -------------------------------------------------------
# State-count figure
# -------------------------------------------------------

state_order = [
    "engaged_like",
    "neutral_like",
    "fatigue_like",
    "frustration_like",
]

state_counts = (
    score_table["proxy_state"]
    .value_counts()
    .reindex(state_order, fill_value=0)
)

fig, ax = plt.subplots(figsize=(9, 5))
state_counts.plot(kind="bar", ax=ax)

ax.set_title(
    "Squat proxy-state counts per participant exercise attempt",
    fontsize=14,
)
ax.set_xlabel("Proxy state")
ax.set_ylabel("Exercise-attempt count")
ax.tick_params(axis="x", rotation=35)

plt.tight_layout()
plt.savefig(COUNT_FIGURE_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved count figure:")
print(COUNT_FIGURE_OUTPUT)


# -------------------------------------------------------
# Console summary
# -------------------------------------------------------

print("\nProxy-state counts:")
print(state_counts)

print("\nPreview:")
print(score_table.head(30).to_string(index=False))

print("\nDone.")