from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\squat_longitudinal_change_dataset.csv"
)

OUTPUT_DIR = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
)

SUMMARY_OUTPUT = OUTPUT_DIR / "attempt_progression_summary.csv"
CHANGE_OUTPUT = OUTPUT_DIR / "attempt1_to_attempt3_changes.csv"
DIRECTION_OUTPUT = OUTPUT_DIR / "attempt1_to_attempt3_direction_summary.csv"


# ============================================================
# LOAD
# ============================================================

data = pd.read_csv(INPUT_PATH)

print("Loaded:")
print(INPUT_PATH)
print("Shape:", data.shape)

data = data.sort_values(
    ["participant", "squat_segment_number"]
).reset_index(drop=True)


# ============================================================
# METRICS
# ============================================================

metrics = [
    # Pose
    "pose_movement",
    "pose_range",
    "pose_stability",

    # Watch
    "watch_acc_mean",
    "watch_acc_variability",
    "watch_gyr_mean",
    "watch_gyr_variability",

    # Heart rate
    "hr_mean",
    "hr_change",
    "hr_slope",
]

for metric in metrics:
    data[metric] = pd.to_numeric(data[metric], errors="coerce")


# ============================================================
# CHECK ATTEMPTS
# ============================================================

print("\n========================================")
print("ATTEMPTS PER PARTICIPANT")
print("========================================")

attempt_counts = (
    data.groupby("participant")["squat_segment_number"]
    .agg(["count", "min", "max"])
)

print(attempt_counts.to_string())

print("\nParticipants:", data["participant"].nunique())
print("Rows:", len(data))

print("\nRows per attempt:")

print(
    data["squat_segment_number"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# GROUP SUMMARY BY ATTEMPT
# ============================================================

summary_rows = []

for metric in metrics:

    for attempt, group in data.groupby("squat_segment_number"):

        values = group[metric].dropna()

        summary_rows.append({
            "metric": metric,
            "attempt": int(attempt),
            "n": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
        })


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


# ============================================================
# ATTEMPT 1 VS ATTEMPT 3
# ============================================================

attempt1 = (
    data[data["squat_segment_number"] == 1]
    .set_index("participant")
)

attempt3 = (
    data[data["squat_segment_number"] == 3]
    .set_index("participant")
)

common_participants = attempt1.index.intersection(
    attempt3.index
)

change_rows = []

for participant in common_participants:

    for metric in metrics:

        value1 = attempt1.loc[participant, metric]
        value3 = attempt3.loc[participant, metric]

        if pd.isna(value1) or pd.isna(value3):
            continue

        delta = value3 - value1

        if abs(value1) > 1e-8:
            pct_change = (delta / abs(value1)) * 100
        else:
            pct_change = np.nan

        if delta > 0:
            direction = "increase"
        elif delta < 0:
            direction = "decrease"
        else:
            direction = "no_change"

        change_rows.append({
            "participant": participant,
            "metric": metric,
            "attempt1_value": value1,
            "attempt3_value": value3,
            "delta": delta,
            "pct_change": pct_change,
            "direction": direction,
        })


changes = pd.DataFrame(change_rows)

changes.to_csv(
    CHANGE_OUTPUT,
    index=False
)


# ============================================================
# HOW CONSISTENT IS EACH DIRECTION?
# ============================================================

direction_rows = []

for metric in metrics:

    metric_changes = changes[
        changes["metric"] == metric
    ]

    increases = (
        metric_changes["direction"] == "increase"
    ).sum()

    decreases = (
        metric_changes["direction"] == "decrease"
    ).sum()

    no_change = (
        metric_changes["direction"] == "no_change"
    ).sum()

    direction_rows.append({
        "metric": metric,
        "participants": len(metric_changes),
        "increased": increases,
        "decreased": decreases,
        "no_change": no_change,
        "median_delta": metric_changes["delta"].median(),
        "mean_delta": metric_changes["delta"].mean(),
        "median_pct_change": metric_changes[
            "pct_change"
        ].median(),
    })


direction_summary = pd.DataFrame(direction_rows)

direction_summary.to_csv(
    DIRECTION_OUTPUT,
    index=False
)


# ============================================================
# PRINT ATTEMPT SUMMARY
# ============================================================

print("\n========================================")
print("MEDIANS BY ATTEMPT")
print("========================================")

median_table = summary.pivot(
    index="metric",
    columns="attempt",
    values="median"
)

print(median_table.round(4).to_string())


# ============================================================
# PRINT ATTEMPT 1 -> 3 DIRECTIONS
# ============================================================

print("\n========================================")
print("ATTEMPT 1 -> ATTEMPT 3")
print("========================================")

print(
    direction_summary.round(4).to_string(index=False)
)


# ============================================================
# HEART RATE SPECIFIC CHECK
# ============================================================

print("\n========================================")
print("HEART RATE ATTEMPT 1 -> 3")
print("========================================")

hr_changes = changes[
    changes["metric"] == "hr_mean"
][
    [
        "participant",
        "attempt1_value",
        "attempt3_value",
        "delta",
        "pct_change",
    ]
]

print(
    hr_changes.round(2).to_string(index=False)
)


# ============================================================
# SAVE INFO
# ============================================================

print("\nSaved:")
print(SUMMARY_OUTPUT)

print("\nSaved:")
print(CHANGE_OUTPUT)

print("\nSaved:")
print(DIRECTION_OUTPUT)

print("\nDone.")
