"""
05_analyse_hr_contribution.py

PRIMARY THESIS ANALYSIS — PHYSIOLOGICAL CONTRIBUTION

Purpose
-------
Examine heart-rate behaviour across repeated exercise attempts while
keeping physiological analysis separate from the pose/wearable analysis.

This script also prepares the matched HR-complete multimodal cohort
required for later fair comparison and fusion experiments.

Research purpose
----------------
Pose and wearable sensors describe movement-related characteristics.
Heart rate represents a different domain: physiological response.

The aim here is therefore NOT to determine whether heart rate is
"better" than pose or wearable sensing.

Instead, the analysis asks:

1. Does heart rate change from A1 to A3?
2. Is the direction of change consistent across workout trajectories?
3. Does heart rate progress through A1 -> A2 -> A3?
4. Does the physiological pattern differ between exercise types?
5. Which trajectories can later be used for a fair matched comparison
   of Pose, Wearable and HR?

Important
---------
Heart-rate change is NOT treated as a direct fatigue label.

An increase in HR does not automatically mean fatigue.
A decrease in HR does not automatically mean improved performance.

The results describe repeated-exercise physiological change only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RESULTS_ROOT = Path(
    r"C:\Masters\results\final_analysis\repeated_performance_analysis"
)

PRIMARY_INPUT = (
    RESULTS_ROOT
    / "data"
    / "01_repeated_attempts_primary_A1_A2_A3.csv"
)

HR_SUMMARY_INPUT = (
    RESULTS_ROOT
    / "tables"
    / "02_hr_feature_change_summary.csv"
)

DATA_OUT = RESULTS_ROOT / "data"
TABLES_OUT = RESULTS_ROOT / "tables"
FIGURES_OUT = RESULTS_ROOT / "figures"
LOGS_OUT = RESULTS_ROOT / "logs"


for folder in [
    DATA_OUT,
    TABLES_OUT,
    FIGURES_OUT,
    LOGS_OUT,
]:

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOAD
# ============================================================

if not PRIMARY_INPUT.exists():

    raise FileNotFoundError(
        "\nPrimary repeated-attempt dataset not found:\n"
        f"{PRIMARY_INPUT}"
    )


if not HR_SUMMARY_INPUT.exists():

    raise FileNotFoundError(
        "\nHR summary from Script 02 not found:\n"
        f"{HR_SUMMARY_INPUT}"
    )


df = pd.read_csv(
    PRIMARY_INPUT
)

hr_summary = pd.read_csv(
    HR_SUMMARY_INPUT
)


print("\n" + "=" * 82)
print("MM-FIT REPEATED PERFORMANCE ANALYSIS")
print("SCRIPT 05 — HEART-RATE PHYSIOLOGICAL CONTRIBUTION")
print("=" * 82)


print(
    f"\nPrimary dataset rows: {len(df)}"
)


print(
    "Primary complete trajectories: "
    f"{df.groupby(['workout_session', 'exercise']).ngroups}"
)


# ============================================================
# IDENTIFY HR-COMPLETE TRAJECTORIES
# ============================================================

# CSV reading can sometimes represent booleans as strings,
# therefore this conversion is deliberately robust.

hr_complete_flag = (
    df[
        "hr_complete_A1_A2_A3"
    ]
    .astype(str)
    .str.lower()
    .eq("true")
)


hr_complete = (
    df.loc[
        hr_complete_flag
    ]
    .copy()
)


# Safety check:
# every retained trajectory must contain exactly three attempts.

trajectory_sizes = (
    hr_complete
    .groupby(
        [
            "workout_session",
            "exercise",
        ]
    )
    .size()
)


if not (
    trajectory_sizes == 3
).all():

    raise RuntimeError(
        "\nHR-complete dataset contains a trajectory "
        "without exactly three attempts."
    )


# Confirm no HR missing values remain.

if (
    hr_complete[
        "hr_mean_combined"
    ]
    .isna()
    .any()
):

    raise RuntimeError(
        "\nUnexpected missing HR value found in "
        "the HR-complete dataset."
    )


n_hr_trajectories = (
    hr_complete
    .groupby(
        [
            "workout_session",
            "exercise",
        ]
    )
    .ngroups
)


print("\nHR-complete matched cohort:")

print(
    f"Rows:                   {len(hr_complete)}"
)

print(
    f"Complete trajectories:  {n_hr_trajectories}"
)

print(
    "True subjects overall: "
    f"{hr_complete['true_subject'].nunique()}"
)

print(
    "Exercises:             "
    f"{hr_complete['exercise'].nunique()}"
)


# ============================================================
# SAVE MATCHED MULTIMODAL COHORT
# ============================================================

# IMPORTANT:
#
# This dataset contains the SAME attempts for:
# - Pose
# - Wearable
# - Heart rate
#
# It will later allow fair modality comparisons and fusion.

MATCHED_FEATURES = [
    # Pose
    "pose_movement_mean",
    "pose_movement_std",
    "pose_movement_max",
    "pose_range_mean",

    # Wearable
    "watch_acc_mag_mean",
    "watch_acc_mag_std",
    "watch_gyr_mag_mean",
    "watch_gyr_mag_std",

    # Physiology
    "hr_mean_combined",
]


MATCHED_IDENTIFIER_COLUMNS = [
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


matched_multimodal = hr_complete[
    MATCHED_IDENTIFIER_COLUMNS
    + MATCHED_FEATURES
].copy()


matched_multimodal.to_csv(
    DATA_OUT
    / "05_hr_complete_matched_multimodal_cohort.csv",
    index=False,
)


# ============================================================
# HR TRAJECTORY TABLE
# ============================================================

wide_hr = (
    hr_complete
    .pivot_table(
        index=[
            "workout_session",
            "true_subject",
            "exercise",
        ],
        columns="attempt",
        values="hr_mean_combined",
        aggfunc="first",
    )
    .reset_index()
)


required_attempts = [
    "A1",
    "A2",
    "A3",
]


for attempt in required_attempts:

    if attempt not in wide_hr.columns:

        raise RuntimeError(
            f"\nMissing expected attempt column: {attempt}"
        )


wide_hr[
    "delta_A1_A2"
] = (
    wide_hr["A2"]
    - wide_hr["A1"]
)


wide_hr[
    "delta_A2_A3"
] = (
    wide_hr["A3"]
    - wide_hr["A2"]
)


wide_hr[
    "delta_A1_A3"
] = (
    wide_hr["A3"]
    - wide_hr["A1"]
)


wide_hr[
    "percent_change_A1_A3"
] = (
    (
        wide_hr["A3"]
        - wide_hr["A1"]
    )
    / wide_hr["A1"]
    * 100.0
)


wide_hr.to_csv(
    DATA_OUT
    / "05_hr_trajectory_changes.csv",
    index=False,
)


# ============================================================
# HR A1-A2-A3 PROGRESSION
# ============================================================

progression_rows = []


for exercise in sorted(
    hr_complete[
        "exercise"
    ].unique()
):

    exercise_df = (
        hr_complete.loc[
            hr_complete[
                "exercise"
            ] == exercise
        ]
        .copy()
    )


    exercise_wide = (
        exercise_df
        .pivot_table(
            index=[
                "workout_session",
                "true_subject",
            ],
            columns="attempt",
            values="hr_mean_combined",
            aggfunc="first",
        )
        .dropna(
            subset=[
                "A1",
                "A2",
                "A3",
            ]
        )
    )


    for attempt_number, attempt in enumerate(
        [
            "A1",
            "A2",
            "A3",
        ],
        start=1,
    ):

        values = (
            exercise_wide[
                attempt
            ]
        )


        paired_change_from_A1 = (
            values
            - exercise_wide["A1"]
        )


        paired_percent_from_A1 = (
            paired_change_from_A1
            / exercise_wide["A1"]
            * 100.0
        )


        progression_rows.append(
            {
                "exercise":
                    exercise,

                "attempt":
                    attempt,

                "attempt_number":
                    attempt_number,

                "n_trajectories":
                    len(values),

                "hr_median_bpm":
                    values.median(),

                "hr_Q1_bpm":
                    values.quantile(
                        0.25
                    ),

                "hr_Q3_bpm":
                    values.quantile(
                        0.75
                    ),

                "median_paired_change_from_A1_bpm":
                    paired_change_from_A1.median(),

                "median_paired_percent_change_from_A1":
                    paired_percent_from_A1.median(),
            }
        )


progression = pd.DataFrame(
    progression_rows
)


progression.to_csv(
    TABLES_OUT
    / "05_hr_progression_summary.csv",
    index=False,
)


# ============================================================
# SECONDARY SUBJECT-LEVEL CHECK
# ============================================================

# Some true subjects completed more than one workout session.
#
# Script 02 uses workout trajectories descriptively.
#
# Here we additionally average repeated workout sessions belonging
# to the same true subject so that each subject contributes once
# per exercise in this secondary check.

subject_attempt = (
    hr_complete
    .groupby(
        [
            "true_subject",
            "exercise",
            "attempt",
        ],
        as_index=False,
    )
    .agg(
        hr_mean_combined=(
            "hr_mean_combined",
            "mean",
        ),
        n_workout_sessions=(
            "workout_session",
            "nunique",
        ),
    )
)


subject_wide = (
    subject_attempt
    .pivot_table(
        index=[
            "true_subject",
            "exercise",
        ],
        columns="attempt",
        values="hr_mean_combined",
        aggfunc="first",
    )
    .reset_index()
)


subject_wide = (
    subject_wide
    .dropna(
        subset=[
            "A1",
            "A2",
            "A3",
        ]
    )
    .copy()
)


subject_wide[
    "delta_A1_A3"
] = (
    subject_wide["A3"]
    - subject_wide["A1"]
)


subject_wide[
    "percent_change_A1_A3"
] = (
    subject_wide[
        "delta_A1_A3"
    ]
    / subject_wide["A1"]
    * 100.0
)


subject_wide.to_csv(
    DATA_OUT
    / "05_hr_subject_level_changes.csv",
    index=False,
)


# ============================================================
# SUBJECT-LEVEL DESCRIPTIVE SUMMARY
# ============================================================

subject_summary_rows = []


for exercise, group in subject_wide.groupby(
    "exercise"
):

    median_delta = (
        group[
            "delta_A1_A3"
        ]
        .median()
    )


    if median_delta > 0:

        direction = "increase"

        consistency = (
            group[
                "delta_A1_A3"
            ] > 0
        ).mean()

    elif median_delta < 0:

        direction = "decrease"

        consistency = (
            group[
                "delta_A1_A3"
            ] < 0
        ).mean()

    else:

        direction = "no_change"

        consistency = (
            np.isclose(
                group[
                    "delta_A1_A3"
                ],
                0,
            )
        ).mean()


    differences = (
        group[
            "delta_A1_A3"
        ]
        .dropna()
    )


    if (
        len(differences) >= 2
        and not np.isclose(
            differences.std(
                ddof=1
            ),
            0,
        )
    ):

        dz = (
            differences.mean()
            / differences.std(
                ddof=1
            )
        )

    else:

        dz = np.nan


    subject_summary_rows.append(
        {
            "exercise":
                exercise,

            "n_true_subjects":
                group[
                    "true_subject"
                ].nunique(),

            "A1_median":
                group[
                    "A1"
                ].median(),

            "A2_median":
                group[
                    "A2"
                ].median(),

            "A3_median":
                group[
                    "A3"
                ].median(),

            "median_delta_A1_A3":
                median_delta,

            "median_percent_change_A1_A3":
                group[
                    "percent_change_A1_A3"
                ].median(),

            "dominant_direction":
                direction,

            "direction_consistency_percent":
                consistency * 100.0,

            "paired_cohens_dz":
                dz,
        }
    )


subject_summary = pd.DataFrame(
    subject_summary_rows
)


subject_summary.to_csv(
    TABLES_OUT
    / "05_hr_subject_level_summary.csv",
    index=False,
)


# ============================================================
# FIGURE 1 — RAW HR PROGRESSION
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        7,
    )
)


for exercise, group in progression.groupby(
    "exercise"
):

    group = group.sort_values(
        "attempt_number"
    )


    ax.plot(
        group[
            "attempt_number"
        ],
        group[
            "hr_median_bpm"
        ],
        marker="o",
        label=exercise.replace(
            "_",
            " ",
        ),
    )


ax.set_xticks(
    [
        1,
        2,
        3,
    ]
)


ax.set_xticklabels(
    [
        "A1",
        "A2",
        "A3",
    ]
)


ax.set_xlabel(
    "Repeated attempt"
)


ax.set_ylabel(
    "Median heart rate (bpm)"
)


ax.set_title(
    "Heart-Rate Response Across Repeated Exercise Attempts"
)


ax.legend(
    bbox_to_anchor=(
        1.02,
        1,
    ),
    loc="upper left",
)


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "05_hr_median_progression.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# FIGURE 2 — PAIRED CHANGE FROM A1
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        12,
        7,
    )
)


for exercise, group in progression.groupby(
    "exercise"
):

    group = group.sort_values(
        "attempt_number"
    )


    ax.plot(
        group[
            "attempt_number"
        ],
        group[
            "median_paired_percent_change_from_A1"
        ],
        marker="o",
        label=exercise.replace(
            "_",
            " ",
        ),
    )


ax.axhline(
    0,
    linewidth=1,
)


ax.set_xticks(
    [
        1,
        2,
        3,
    ]
)


ax.set_xticklabels(
    [
        "A1",
        "A2",
        "A3",
    ]
)


ax.set_xlabel(
    "Repeated attempt"
)


ax.set_ylabel(
    "Median within-trajectory HR change from A1 (%)"
)


ax.set_title(
    "Paired Heart-Rate Change Relative to A1"
)


ax.legend(
    bbox_to_anchor=(
        1.02,
        1,
    ),
    loc="upper left",
)


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "05_hr_paired_percent_progression.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# MATCHED COHORT SUMMARY
# ============================================================

matched_cohort_summary = (
    hr_complete
    .groupby(
        "exercise",
        as_index=False,
    )
    .agg(
        n_rows=(
            "segment_id",
            "size",
        ),

        n_complete_trajectories=(
            "workout_session",
            "nunique",
        ),

        n_true_subjects=(
            "true_subject",
            "nunique",
        ),
    )
)


# Because one workout session contributes one complete trajectory
# for a given exercise in this dataset, n_workout_sessions equals
# n_complete_trajectories here.

matched_cohort_summary.to_csv(
    TABLES_OUT
    / "05_matched_multimodal_cohort_summary.csv",
    index=False,
)


# ============================================================
# METHOD RECORD
# ============================================================

method_record = """
SCRIPT 05 — HEART-RATE PHYSIOLOGICAL CONTRIBUTION

Purpose:
Evaluate heart-rate change across repeated exercise attempts while
preserving heart rate as a separate physiological modality.

Primary HR cohort:
Only workout/exercise trajectories containing heart-rate data at
A1, A2 and A3.

The pose and wearable datasets are NOT reduced globally.
The HR-complete subset is created separately.

Primary interpretation:
Heart-rate change represents physiological response across repeated
exercise performance.

Heart rate is NOT treated as:
- a fatigue label,
- an engagement label,
- a deterioration label,
- a clinical outcome.

Metrics:
- A1, A2 and A3 heart-rate medians
- paired A1-to-A3 change in bpm
- paired percentage change
- directional consistency
- A1-to-A2-to-A3 progression
- paired Cohen's dz

Secondary subject-level check:
Some MM-Fit subjects completed multiple workout sessions.
A secondary summary therefore averages repeated workout sessions
within true subject + exercise + attempt before calculating A1-to-A3
change. This prevents subjects with multiple sessions from contributing
multiple times to that secondary descriptive check.

Matched multimodal cohort:
A separate dataset is created containing only HR-complete trajectories,
while retaining all four pose features, all four wearable features,
and heart rate.

This matched dataset will later support a fair comparison of:
- Pose
- Wearable
- HR
- Pose + Wearable
- Pose + Wearable + HR

Fusion interpretation:
Fusion will operate on the original attempt-level sensor features.
Cohen's dz values are NOT fused and no artificial multimodal sensitivity
score is created.
"""


(
    LOGS_OUT
    / "05_method_record.txt"
).write_text(
    method_record.strip(),
    encoding="utf-8",
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n" + "-" * 82)
print("HR-COMPLETE COHORT BY EXERCISE")
print("-" * 82)


print(
    matched_cohort_summary.to_string(
        index=False
    )
)


print("\n" + "-" * 82)
print("TRAJECTORY-LEVEL HR RESULTS FROM SCRIPT 02")
print("-" * 82)


trajectory_display_columns = [
    "exercise",
    "n_trajectories",
    "A1_median",
    "A2_median",
    "A3_median",
    "median_delta_A1_A3",
    "median_percent_change_A1_A3",
    "dominant_direction",
    "endpoint_direction_consistency_percent",
    "monotonic_A1_A2_A3_consistency_percent",
    "paired_cohens_dz",
]


print(
    hr_summary[
        trajectory_display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 82)
print("SECONDARY TRUE-SUBJECT-LEVEL HR CHECK")
print("-" * 82)


print(
    subject_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "=" * 82)
print("SCRIPT 05 COMPLETE")
print("=" * 82)


print(
    "\nMatched multimodal dataset:"
)


print(
    DATA_OUT
    / "05_hr_complete_matched_multimodal_cohort.csv"
)


print(
    "\nHR figures saved to:"
)


print(
    FIGURES_OUT
)


print(
    "\nHeart rate remains physiologically interpreted "
    "and has NOT been treated as a fatigue label."
)

print(
    "\nThe matched dataset is now ready for the later "
    "Pose / Wearable / HR comparison and fusion analysis."
)