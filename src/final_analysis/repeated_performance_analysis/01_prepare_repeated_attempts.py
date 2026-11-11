"""
01_prepare_repeated_attempts.py

PRIMARY THESIS ANALYSIS — DATASET PREPARATION

Purpose
-------
Prepare a clean repeated-performance dataset from the MM-Fit master
segment table.

This script:

1. Loads the MM-Fit master segment table.
2. Converts workout IDs (w00-w20) to the correct 10 true subjects.
3. Creates the simplified bilateral smartwatch features.
4. Orders each exercise chronologically within each workout session.
5. Assigns repeated attempts A1, A2, A3, etc.
6. Identifies complete A1-A2-A3 trajectories.
7. Retains the first three attempts for the primary repeated-performance analysis.
8. Records HR availability separately.
9. Creates all result folders automatically.
10. Saves QC tables and a feature-definition register.

IMPORTANT
---------
A1, A2 and A3 are repeated exercise attempts.

They are NOT:
- fatigue labels
- engagement labels
- behavioural-state labels

No statistical testing is performed in this script.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    r"C:\Masters\data\features\mmfit\mmfit_master_segment_table.csv"
)

RESULTS_ROOT = Path(
    r"C:\Masters\results\final_analysis\repeated_performance_analysis"
)

DATA_OUT = RESULTS_ROOT / "data"
TABLES_OUT = RESULTS_ROOT / "tables"
FIGURES_OUT = RESULTS_ROOT / "figures"
LOGS_OUT = RESULTS_ROOT / "logs"


# Create result folders automatically
for folder in [
    RESULTS_ROOT,
    DATA_OUT,
    TABLES_OUT,
    FIGURES_OUT,
    LOGS_OUT,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRUE SUBJECT MAPPING
# ============================================================

# MM-Fit contains 21 workout sessions but only 10 true subjects.
#
# The original master table uses values such as w00, w01, etc.
# These identify workout sessions, NOT individual participants.

WORKOUT_TO_SUBJECT = {
    "w00": 2,
    "w01": 0,
    "w02": 1,
    "w03": 0,
    "w04": 1,
    "w05": 2,
    "w06": 0,
    "w07": 1,
    "w08": 0,
    "w09": 1,
    "w10": 0,
    "w11": 1,
    "w12": 3,
    "w13": 4,
    "w14": 0,
    "w15": 1,
    "w16": 5,
    "w17": 6,
    "w18": 7,
    "w19": 8,
    "w20": 9,
}


# ============================================================
# PRIMARY FEATURE SET
# ============================================================

# POSE
#
# These deliberately summarise whole-body movement rather than
# using a very large collection of joint-specific features.

POSE_FEATURES = [
    "pose_movement_mean",
    "pose_movement_std",
    "pose_movement_max",
    "pose_range_mean",
]


# WATCH SOURCE COLUMNS
#
# MM-Fit provides left and right smartwatch measurements.
# We combine corresponding left/right measurements into simple
# bilateral summary features.

WATCH_SOURCE_COLUMNS = [
    "sw_l_acc_mag_mean",
    "sw_r_acc_mag_mean",
    "sw_l_acc_mag_std",
    "sw_r_acc_mag_std",
    "sw_l_gyr_mag_mean",
    "sw_r_gyr_mag_mean",
    "sw_l_gyr_mag_std",
    "sw_r_gyr_mag_std",
]


# PHYSIOLOGY

PHYSIOLOGY_FEATURES = [
    "hr_mean_combined",
]


# Columns needed to identify and order exercise segments

IDENTIFIER_COLUMNS = [
    "segment_id",
    "participant",
    "participant_segment_number",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
]


REQUIRED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + POSE_FEATURES
    + WATCH_SOURCE_COLUMNS
    + PHYSIOLOGY_FEATURES
)


# ============================================================
# LOAD MASTER TABLE
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "\nMaster table could not be found.\n\n"
        f"Expected location:\n{INPUT_PATH}\n"
    )


df = pd.read_csv(INPUT_PATH)


print("\n" + "=" * 75)
print("MM-FIT REPEATED PERFORMANCE ANALYSIS")
print("SCRIPT 01 — PREPARE REPEATED ATTEMPTS")
print("=" * 75)

print(f"\nInput file:\n{INPUT_PATH}")

print("\nOriginal dataset:")
print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:

    print("\nMissing required columns:")

    for column in missing_columns:
        print(f"  - {column}")

    raise ValueError(
        "\nThe master table does not contain all columns required "
        "for the repeated-performance analysis."
    )


# ============================================================
# WORKOUT SESSION AND TRUE SUBJECT
# ============================================================

df = df.copy()


# Keep the original participant field but give it its correct
# methodological name.

df["workout_session"] = df["participant"].astype(str)


# Add true MM-Fit subject

df["true_subject"] = df["workout_session"].map(
    WORKOUT_TO_SUBJECT
)


# Stop immediately if an unknown workout ID appears.

if df["true_subject"].isna().any():

    unmapped = sorted(
        df.loc[
            df["true_subject"].isna(),
            "workout_session"
        ].unique()
    )

    raise ValueError(
        "\nThe following workout IDs have no true-subject mapping:\n"
        + "\n".join(unmapped)
    )


df["true_subject"] = df["true_subject"].astype(int)


print("\nDataset identity check:")
print(
    f"Workout sessions: "
    f"{df['workout_session'].nunique()}"
)

print(
    f"True subjects:     "
    f"{df['true_subject'].nunique()}"
)

print(
    f"Exercise types:    "
    f"{df['exercise'].nunique()}"
)


# ============================================================
# CREATE SIMPLE BILATERAL WATCH FEATURES
# ============================================================

# Accelerometer magnitude — average movement intensity

df["watch_acc_mag_mean"] = df[
    [
        "sw_l_acc_mag_mean",
        "sw_r_acc_mag_mean",
    ]
].mean(axis=1)


# Accelerometer magnitude — average movement variability

df["watch_acc_mag_std"] = df[
    [
        "sw_l_acc_mag_std",
        "sw_r_acc_mag_std",
    ]
].mean(axis=1)


# Gyroscope magnitude — average rotational movement

df["watch_gyr_mag_mean"] = df[
    [
        "sw_l_gyr_mag_mean",
        "sw_r_gyr_mag_mean",
    ]
].mean(axis=1)


# Gyroscope magnitude — average rotational variability

df["watch_gyr_mag_std"] = df[
    [
        "sw_l_gyr_mag_std",
        "sw_r_gyr_mag_std",
    ]
].mean(axis=1)


WATCH_FEATURES = [
    "watch_acc_mag_mean",
    "watch_acc_mag_std",
    "watch_gyr_mag_mean",
    "watch_gyr_mag_std",
]


CORE_FEATURES = (
    POSE_FEATURES
    + WATCH_FEATURES
    + PHYSIOLOGY_FEATURES
)


# ============================================================
# ASSIGN REPEATED ATTEMPTS
# ============================================================

# The attempt number must be assigned separately for each:
#
#     workout session + exercise
#
# The segments are ordered by their original start time.
#
# participant_segment_number is NOT used to determine A1/A2/A3.


df = df.sort_values(
    [
        "workout_session",
        "exercise",
        "start_time",
        "segment_id",
    ]
).reset_index(drop=True)


df["attempt_number"] = (
    df
    .groupby(
        [
            "workout_session",
            "exercise",
        ]
    )
    .cumcount()
    + 1
)


df["attempt"] = (
    "A"
    + df["attempt_number"].astype(str)
)


# ============================================================
# ATTEMPT COUNT QC
# ============================================================

attempt_qc = (
    df
    .groupby(
        [
            "workout_session",
            "true_subject",
            "exercise",
        ],
        as_index=False,
    )
    .agg(
        n_attempts=(
            "attempt_number",
            "max",
        ),
        first_start_time=(
            "start_time",
            "min",
        ),
        last_start_time=(
            "start_time",
            "max",
        ),
    )
)


attempt_qc["complete_A1_A2_A3"] = (
    attempt_qc["n_attempts"] >= 3
)


attempt_qc["has_attempt_after_A3"] = (
    attempt_qc["n_attempts"] > 3
)


attempt_qc.to_csv(
    TABLES_OUT / "01_attempt_count_qc.csv",
    index=False,
)


# ============================================================
# PRIMARY A1-A2-A3 COHORT
# ============================================================

# Identify workout/exercise combinations that contain at least
# three chronological attempts.

complete_groups = attempt_qc.loc[
    attempt_qc["complete_A1_A2_A3"],
    [
        "workout_session",
        "exercise",
    ],
].copy()


primary = df.merge(
    complete_groups,
    on=[
        "workout_session",
        "exercise",
    ],
    how="inner",
)


# Standardise the primary analysis to A1/A2/A3.
#
# If a workout contains an A4, the A4 remains documented in the
# full dataset and QC table but is not part of the primary analysis.

primary = primary.loc[
    primary["attempt_number"] <= 3
].copy()


# Safety check:
# every retained workout/exercise trajectory must contain exactly
# three rows.

trajectory_sizes = (
    primary
    .groupby(
        [
            "workout_session",
            "exercise",
        ]
    )
    .size()
)


if not (trajectory_sizes == 3).all():

    raise RuntimeError(
        "\nUnexpected trajectory size detected. "
        "Every primary trajectory should contain exactly "
        "A1, A2 and A3."
    )


# ============================================================
# HEART RATE AVAILABILITY
# ============================================================

# HR has some missing segments.
#
# We DO NOT discard valid pose/watch trajectories simply because
# HR is unavailable.
#
# Instead, HR-complete trajectories are recorded separately.

primary["hr_available"] = (
    primary["hr_mean_combined"].notna()
)


hr_complete = (
    primary
    .groupby(
        [
            "workout_session",
            "exercise",
        ]
    )["hr_available"]
    .all()
    .rename(
        "hr_complete_A1_A2_A3"
    )
    .reset_index()
)


primary = primary.merge(
    hr_complete,
    on=[
        "workout_session",
        "exercise",
    ],
    how="left",
)


# ============================================================
# SAVE ANALYSIS DATASETS
# ============================================================

BASE_OUTPUT_COLUMNS = [
    "segment_id",
    "workout_session",
    "true_subject",
    "exercise",
    "attempt_number",
    "attempt",
    "start_time",
    "end_time",
    "duration",
    "reps",
]


FULL_OUTPUT_COLUMNS = (
    BASE_OUTPUT_COLUMNS
    + CORE_FEATURES
)


PRIMARY_OUTPUT_COLUMNS = (
    BASE_OUTPUT_COLUMNS
    + CORE_FEATURES
    + [
        "hr_available",
        "hr_complete_A1_A2_A3",
    ]
)


df[FULL_OUTPUT_COLUMNS].to_csv(
    DATA_OUT / "01_repeated_attempts_all.csv",
    index=False,
)


primary[PRIMARY_OUTPUT_COLUMNS].to_csv(
    DATA_OUT / "01_repeated_attempts_primary_A1_A2_A3.csv",
    index=False,
)


# ============================================================
# EXERCISE COHORT SUMMARY
# ============================================================

summary_rows = []


for exercise, group in primary.groupby("exercise"):

    session_subjects = (
        group[
            [
                "workout_session",
                "true_subject",
            ]
        ]
        .drop_duplicates()
    )


    hr_by_session = (
        group
        .groupby(
            "workout_session"
        )["hr_complete_A1_A2_A3"]
        .first()
    )


    summary_rows.append(
        {
            "exercise": exercise,

            "complete_A1_A2_A3_workout_sessions":
                session_subjects[
                    "workout_session"
                ].nunique(),

            "true_subjects_represented":
                session_subjects[
                    "true_subject"
                ].nunique(),

            "primary_rows":
                len(group),

            "hr_complete_A1_A2_A3_workout_sessions":
                int(hr_by_session.sum()),
        }
    )


exercise_summary = pd.DataFrame(
    summary_rows
)


exercise_summary = exercise_summary.sort_values(
    [
        "complete_A1_A2_A3_workout_sessions",
        "exercise",
    ],
    ascending=[
        False,
        True,
    ],
)


exercise_summary.to_csv(
    TABLES_OUT / "01_exercise_cohort_summary.csv",
    index=False,
)


# ============================================================
# FEATURE DEFINITION REGISTER
# ============================================================

feature_register = pd.DataFrame(
    [
        {
            "feature": "pose_movement_mean",
            "modality": "Pose",
            "source_or_formula": "pose_movement_mean",
            "interpretation":
                "Average whole-body frame-to-frame skeletal movement",
        },

        {
            "feature": "pose_movement_std",
            "modality": "Pose",
            "source_or_formula": "pose_movement_std",
            "interpretation":
                "Variability in whole-body skeletal movement",
        },

        {
            "feature": "pose_movement_max",
            "modality": "Pose",
            "source_or_formula": "pose_movement_max",
            "interpretation":
                "Maximum observed whole-body skeletal movement",
        },

        {
            "feature": "pose_range_mean",
            "modality": "Pose",
            "source_or_formula": "pose_range_mean",
            "interpretation":
                "Average movement range across pose joints",
        },

        {
            "feature": "watch_acc_mag_mean",
            "modality": "Wearable",
            "source_or_formula":
                "(sw_l_acc_mag_mean + sw_r_acc_mag_mean) / 2",
            "interpretation":
                "Average bilateral wrist acceleration magnitude",
        },

        {
            "feature": "watch_acc_mag_std",
            "modality": "Wearable",
            "source_or_formula":
                "(sw_l_acc_mag_std + sw_r_acc_mag_std) / 2",
            "interpretation":
                "Average bilateral variability in wrist acceleration magnitude",
        },

        {
            "feature": "watch_gyr_mag_mean",
            "modality": "Wearable",
            "source_or_formula":
                "(sw_l_gyr_mag_mean + sw_r_gyr_mag_mean) / 2",
            "interpretation":
                "Average bilateral wrist rotational movement",
        },

        {
            "feature": "watch_gyr_mag_std",
            "modality": "Wearable",
            "source_or_formula":
                "(sw_l_gyr_mag_std + sw_r_gyr_mag_std) / 2",
            "interpretation":
                "Average bilateral variability in wrist rotational movement",
        },

        {
            "feature": "hr_mean_combined",
            "modality": "Physiological",
            "source_or_formula": "hr_mean_combined",
            "interpretation":
                "Mean combined heart rate during the exercise segment",
        },
    ]
)


feature_register.to_csv(
    TABLES_OUT / "01_feature_register.csv",
    index=False,
)


# ============================================================
# SAVE TRUE SUBJECT MAPPING
# ============================================================

subject_mapping = pd.DataFrame(
    sorted(
        WORKOUT_TO_SUBJECT.items()
    ),
    columns=[
        "workout_session",
        "true_subject",
    ],
)


subject_mapping.to_csv(
    TABLES_OUT / "01_workout_to_true_subject_mapping.csv",
    index=False,
)


# ============================================================
# TERMINAL QC OUTPUT
# ============================================================

print("\n" + "-" * 75)
print("PRIMARY FEATURE AVAILABILITY")
print("-" * 75)


for feature in CORE_FEATURES:

    missing = primary[
        feature
    ].isna().sum()

    print(
        f"{feature:<25} "
        f"missing rows: {missing}"
    )


print("\n" + "-" * 75)
print("PRIMARY A1-A2-A3 COHORT")
print("-" * 75)

print(
    exercise_summary.to_string(
        index=False
    )
)


# ============================================================
# EXERCISES EXCLUDED FROM PRIMARY ANALYSIS
# ============================================================

all_exercises = set(
    df["exercise"].unique()
)

primary_exercises = set(
    primary["exercise"].unique()
)

excluded_exercises = sorted(
    all_exercises
    - primary_exercises
)


print("\n" + "-" * 75)
print("EXERCISES WITHOUT COMPLETE A1-A2-A3 TRAJECTORIES")
print("-" * 75)


if excluded_exercises:

    for exercise in excluded_exercises:
        print(exercise)

else:

    print("None")


# ============================================================
# EXTRA ATTEMPTS
# ============================================================

extra_attempts = attempt_qc.loc[
    attempt_qc[
        "has_attempt_after_A3"
    ]
]


print("\n" + "-" * 75)
print("WORKOUT/EXERCISE GROUPS WITH MORE THAN THREE ATTEMPTS")
print("-" * 75)


if len(extra_attempts) == 0:

    print("None")

else:

    print(
        extra_attempts[
            [
                "workout_session",
                "true_subject",
                "exercise",
                "n_attempts",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("SCRIPT 01 COMPLETE")
print("=" * 75)

print(
    f"\nPrimary repeated-performance rows: "
    f"{len(primary)}"
)

print(
    f"Complete workout/exercise trajectories: "
    f"{primary.groupby(['workout_session', 'exercise']).ngroups}"
)

print(
    f"True subjects represented: "
    f"{primary['true_subject'].nunique()}"
)


print("\nResults saved automatically to:")

print(
    r"C:\Masters\results\final_analysis"
    r"\repeated_performance_analysis"
)


print("\nNo feature ranking or statistical testing has been performed.")
print("The next stage is Script 02: repeated-performance change analysis.")