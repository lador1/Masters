from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    r"C:\Masters\results\mmfit\squat_metrics_by_segment_usable_only.csv"
)

OUTPUT_DIR = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "squat_longitudinal_change_dataset.csv"
ELIGIBILITY_PATH = OUTPUT_DIR / "squat_participant_attempt_summary.csv"


# ============================================================
# LOAD
# ============================================================

data = pd.read_csv(INPUT_PATH)

print("Loaded:")
print(INPUT_PATH)
print("Shape:", data.shape)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "squat_segment_id",
    "participant",
    "squat_segment_number",

    # Pose
    "pose_movement",
    "pose_range",
    "pose_stability",

    # Watch
    "watch_acc_mean",
    "watch_acc_variability",
    "watch_gyr_mean",
    "watch_gyr_variability",

    # Physiological effort
    "hr_mean",
    "hr_change",
    "hr_slope",
]

missing = [c for c in required_columns if c not in data.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")


# ============================================================
# CLEAN / SORT
# ============================================================

numeric_columns = [
    "squat_segment_number",
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

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan)

data = data.dropna(
    subset=[
        "participant",
        "squat_segment_number",
        "pose_movement",
        "pose_range",
        "pose_stability",
        "watch_acc_mean",
        "watch_acc_variability",
        "watch_gyr_mean",
        "watch_gyr_variability",
        "hr_mean",
    ]
).copy()

data = data.sort_values(
    ["participant", "squat_segment_number"]
).reset_index(drop=True)


# ============================================================
# PARTICIPANT ATTEMPT SUMMARY
# ============================================================

summary_rows = []

for participant, group in data.groupby("participant"):

    attempts = sorted(
        group["squat_segment_number"]
        .astype(int)
        .unique()
        .tolist()
    )

    summary_rows.append({
        "participant": participant,
        "usable_attempt_count": len(attempts),
        "attempt_numbers": ",".join(map(str, attempts)),
        "has_attempt_1": int(1 in attempts),
        "has_3_or_more_attempts": int(len(attempts) >= 3),
        "has_4_or_more_attempts": int(len(attempts) >= 4),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(ELIGIBILITY_PATH, index=False)


# ============================================================
# IMPORTANT:
# Only participants with a true Attempt 1 can use Attempt 1
# as their personal baseline.
# ============================================================

eligible_participants = summary.loc[
    (summary["has_attempt_1"] == 1)
    & (summary["has_3_or_more_attempts"] == 1),
    "participant"
]

longitudinal = data[
    data["participant"].isin(eligible_participants)
].copy()


# ============================================================
# FEATURES TO TRACK OVER TIME
# ============================================================

feature_columns = [
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


# ============================================================
# ATTEMPT 1 PERSONAL BASELINE
# ============================================================

baseline = (
    longitudinal[
        longitudinal["squat_segment_number"] == 1
    ]
    .set_index("participant")[feature_columns]
)

baseline = baseline.add_suffix("_baseline")

longitudinal = longitudinal.merge(
    baseline,
    left_on="participant",
    right_index=True,
    how="inner",
)


# ============================================================
# CHANGE FROM PERSONAL BASELINE
#
# delta:
#     current value - Attempt 1 value
#
# pct_change:
#     percentage change relative to Attempt 1
# ============================================================

EPSILON = 1e-8

for feature in feature_columns:

    baseline_col = f"{feature}_baseline"

    longitudinal[f"{feature}_delta"] = (
        longitudinal[feature]
        - longitudinal[baseline_col]
    )

    denominator = longitudinal[baseline_col].abs()

    longitudinal[f"{feature}_pct_change"] = np.where(
        denominator > EPSILON,
        (
            longitudinal[f"{feature}_delta"]
            / denominator
        ) * 100.0,
        np.nan,
    )


# ============================================================
# TIME / ATTEMPT POSITION
# ============================================================

longitudinal["attempt_from_baseline"] = (
    longitudinal["squat_segment_number"] - 1
)


# ============================================================
# SAVE
# ============================================================

longitudinal = longitudinal.sort_values(
    ["participant", "squat_segment_number"]
)

longitudinal.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n----------------------------------------")
print("PARTICIPANT ATTEMPT SUMMARY")
print("----------------------------------------")

print(
    summary.to_string(index=False)
)

print("\nParticipants total:")
print(summary["participant"].nunique())

print("\nParticipants with true Attempt 1:")
print(summary["has_attempt_1"].sum())

print("\nParticipants with >= 3 usable attempts:")
print(summary["has_3_or_more_attempts"].sum())

print("\nParticipants eligible for longitudinal analysis:")
print(longitudinal["participant"].nunique())

print("\nRows in longitudinal dataset:")
print(len(longitudinal))

print("\nAttempts retained:")
print(
    longitudinal.groupby("squat_segment_number")
    .size()
    .rename("rows")
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nSaved participant summary:")
print(ELIGIBILITY_PATH)


# ============================================================
# QUICK PREVIEW OF CHANGES
# ============================================================

preview_columns = [
    "participant",
    "squat_segment_number",

    "pose_range",
    "pose_range_pct_change",

    "pose_stability",
    "pose_stability_pct_change",

    "watch_acc_variability",
    "watch_acc_variability_pct_change",

    "hr_mean",
    "hr_mean_pct_change",
]

print("\nPreview:")
print(
    longitudinal[
        preview_columns
    ].head(40).to_string(index=False)
)

print("\nDone")