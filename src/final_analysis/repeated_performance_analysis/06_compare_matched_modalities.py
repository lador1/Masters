"""
06_compare_matched_modalities.py

PRIMARY THESIS ANALYSIS — MATCHED THREE-MODALITY COMPARISON

Purpose
-------
Compare Pose, Wearable and Heart Rate using exactly the same
HR-complete A1/A2/A3 trajectories.

Why this script is necessary
----------------------------
The main Pose/Watch analysis used 181 complete trajectories.

Heart rate is only available for 154 complete trajectories.

Therefore it would be unfair to directly compare:
    Pose/Watch results from 181 trajectories
with:
    HR results from 154 trajectories.

This script recalculates ALL modalities using the same 154
HR-complete trajectories.

Modalities
----------
POSE:
    4 frozen pose-derived features

WEARABLE:
    4 frozen smartwatch-derived features

PHYSIOLOGICAL:
    1 heart-rate feature

Important limitation
--------------------
Pose and Wearable are represented by four features each.
Heart Rate is represented by one feature.

Therefore:
- this is a descriptive comparison of responsiveness,
- it is NOT a competition,
- it does NOT prove one modality is "better",
- HR should not be interpreted as directly equivalent to a
  four-feature modality summary.

Fusion is NOT performed in this script.

Fusion will later operate on the original attempt-level features.
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

INPUT_PATH = (
    RESULTS_ROOT
    / "data"
    / "05_hr_complete_matched_multimodal_cohort.csv"
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
# FROZEN FEATURES
# ============================================================

POSE_FEATURES = [
    "pose_movement_mean",
    "pose_movement_std",
    "pose_movement_max",
    "pose_range_mean",
]


WATCH_FEATURES = [
    "watch_acc_mag_mean",
    "watch_acc_mag_std",
    "watch_gyr_mag_mean",
    "watch_gyr_mag_std",
]


HR_FEATURES = [
    "hr_mean_combined",
]


ALL_FEATURES = (
    POSE_FEATURES
    + WATCH_FEATURES
    + HR_FEATURES
)


FEATURE_MODALITY = {
    "pose_movement_mean": "Pose",
    "pose_movement_std": "Pose",
    "pose_movement_max": "Pose",
    "pose_range_mean": "Pose",

    "watch_acc_mag_mean": "Wearable",
    "watch_acc_mag_std": "Wearable",
    "watch_gyr_mag_mean": "Wearable",
    "watch_gyr_mag_std": "Wearable",

    "hr_mean_combined": "Physiological",
}


# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        "\nMatched HR-complete multimodal cohort not found:\n"
        f"{INPUT_PATH}"
    )


df = pd.read_csv(
    INPUT_PATH
)


print("\n" + "=" * 84)
print("MM-FIT REPEATED PERFORMANCE ANALYSIS")
print("SCRIPT 06 — MATCHED POSE / WEARABLE / HR COMPARISON")
print("=" * 84)


print(
    f"\nInput:\n{INPUT_PATH}"
)


print(
    f"\nRows: {len(df)}"
)


n_trajectories = (
    df
    .groupby(
        [
            "workout_session",
            "exercise",
        ]
    )
    .ngroups
)


print(
    f"Complete trajectories: {n_trajectories}"
)


print(
    f"True subjects: {df['true_subject'].nunique()}"
)


print(
    f"Exercises: {df['exercise'].nunique()}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "workout_session",
    "true_subject",
    "exercise",
    "attempt",
] + ALL_FEATURES


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# Every trajectory must have A1/A2/A3.

trajectory_sizes = (
    df
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
        "\nMatched cohort contains trajectories "
        "without exactly three attempts."
    )


# All nine features must be complete in matched cohort.

if df[
    ALL_FEATURES
].isna().any().any():

    raise RuntimeError(
        "\nMatched multimodal cohort contains missing "
        "sensor feature values."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def paired_dz(
    differences
):

    differences = (
        pd.Series(
            differences
        )
        .dropna()
        .astype(float)
    )


    if len(
        differences
    ) < 2:

        return np.nan


    sd = (
        differences.std(
            ddof=1
        )
    )


    if np.isclose(
        sd,
        0,
    ):

        return np.nan


    return (
        differences.mean()
        / sd
    )


def dominant_direction(
    median_change
):

    if median_change > 0:

        return "increase"

    elif median_change < 0:

        return "decrease"

    else:

        return "no_change"


# ============================================================
# FEATURE-LEVEL MATCHED ANALYSIS
# ============================================================

feature_rows = []


for exercise in sorted(
    df[
        "exercise"
    ].unique()
):

    exercise_df = (
        df.loc[
            df[
                "exercise"
            ] == exercise
        ]
        .copy()
    )


    for feature in ALL_FEATURES:

        modality = (
            FEATURE_MODALITY[
                feature
            ]
        )


        wide = (
            exercise_df
            .pivot_table(
                index=[
                    "workout_session",
                    "true_subject",
                ],
                columns="attempt",
                values=feature,
                aggfunc="first",
            )
            .dropna(
                subset=[
                    "A1",
                    "A2",
                    "A3",
                ]
            )
            .copy()
        )


        wide[
            "delta_A1_A3"
        ] = (
            wide["A3"]
            - wide["A1"]
        )


        wide[
            "percent_change_A1_A3"
        ] = np.where(

            ~np.isclose(
                wide["A1"],
                0,
            ),

            (
                wide[
                    "delta_A1_A3"
                ]
                / wide["A1"]
                * 100.0
            ),

            np.nan,
        )


        median_delta = (
            wide[
                "delta_A1_A3"
            ]
            .median()
        )


        direction = (
            dominant_direction(
                median_delta
            )
        )


        if direction == "increase":

            endpoint_consistent = (
                wide["A3"]
                > wide["A1"]
            )


            monotonic_consistent = (
                (
                    wide["A2"]
                    >= wide["A1"]
                )
                &
                (
                    wide["A3"]
                    >= wide["A2"]
                )
                &
                (
                    wide["A3"]
                    > wide["A1"]
                )
            )


        elif direction == "decrease":

            endpoint_consistent = (
                wide["A3"]
                < wide["A1"]
            )


            monotonic_consistent = (
                (
                    wide["A2"]
                    <= wide["A1"]
                )
                &
                (
                    wide["A3"]
                    <= wide["A2"]
                )
                &
                (
                    wide["A3"]
                    < wide["A1"]
                )
            )


        else:

            endpoint_consistent = (
                np.isclose(
                    wide["A3"],
                    wide["A1"],
                )
            )


            monotonic_consistent = (
                np.isclose(
                    wide["A1"],
                    wide["A2"],
                )
                &
                np.isclose(
                    wide["A2"],
                    wide["A3"],
                )
            )


        feature_rows.append(
            {
                "exercise":
                    exercise,

                "feature":
                    feature,

                "modality":
                    modality,

                "n_trajectories":
                    len(
                        wide
                    ),

                "n_true_subjects":
                    wide
                    .reset_index()[
                        "true_subject"
                    ]
                    .nunique(),

                "A1_median":
                    wide[
                        "A1"
                    ].median(),

                "A2_median":
                    wide[
                        "A2"
                    ].median(),

                "A3_median":
                    wide[
                        "A3"
                    ].median(),

                "median_delta_A1_A3":
                    median_delta,

                "median_percent_change_A1_A3":
                    wide[
                        "percent_change_A1_A3"
                    ]
                    .median(),

                "dominant_direction":
                    direction,

                "endpoint_direction_consistency_percent":
                    endpoint_consistent.mean()
                    * 100.0,

                "monotonic_A1_A2_A3_consistency_percent":
                    monotonic_consistent.mean()
                    * 100.0,

                "paired_cohens_dz":
                    paired_dz(
                        wide[
                            "delta_A1_A3"
                        ]
                    ),
            }
        )


feature_summary = pd.DataFrame(
    feature_rows
)


feature_summary[
    "absolute_dz"
] = (
    feature_summary[
        "paired_cohens_dz"
    ].abs()
)


feature_summary[
    "absolute_median_percent_change"
] = (
    feature_summary[
        "median_percent_change_A1_A3"
    ].abs()
)


feature_summary.to_csv(
    TABLES_OUT
    / "06_matched_feature_change_summary.csv",
    index=False,
)


# ============================================================
# EXERCISE x MODALITY SUMMARY
# ============================================================

# Pose and Wearable:
# median across their four frozen features.
#
# Physiological:
# one HR feature, therefore its modality value is simply the HR result.

modality_rows = []


for (
    exercise,
    modality
), group in feature_summary.groupby(
    [
        "exercise",
        "modality",
    ]
):

    modality_rows.append(
        {
            "exercise":
                exercise,

            "modality":
                modality,

            "n_features":
                group[
                    "feature"
                ].nunique(),

            "n_trajectories":
                group[
                    "n_trajectories"
                ].iloc[0],

            "n_true_subjects":
                group[
                    "n_true_subjects"
                ].iloc[0],

            "median_absolute_dz":
                group[
                    "absolute_dz"
                ].median(),

            "mean_absolute_dz":
                group[
                    "absolute_dz"
                ].mean(),

            "median_endpoint_consistency_percent":
                group[
                    "endpoint_direction_consistency_percent"
                ].median(),

            "median_monotonic_consistency_percent":
                group[
                    "monotonic_A1_A2_A3_consistency_percent"
                ].median(),

            "median_absolute_percent_change":
                group[
                    "absolute_median_percent_change"
                ].median(),
        }
    )


exercise_modality_summary = pd.DataFrame(
    modality_rows
)


exercise_modality_summary.to_csv(
    TABLES_OUT
    / "06_matched_exercise_modality_summary.csv",
    index=False,
)


# ============================================================
# WIDE COMPARISON TABLE
# ============================================================

dz_wide = (
    exercise_modality_summary
    .pivot(
        index="exercise",
        columns="modality",
        values="median_absolute_dz",
    )
)


consistency_wide = (
    exercise_modality_summary
    .pivot(
        index="exercise",
        columns="modality",
        values="median_endpoint_consistency_percent",
    )
)


monotonic_wide = (
    exercise_modality_summary
    .pivot(
        index="exercise",
        columns="modality",
        values="median_monotonic_consistency_percent",
    )
)


comparison_rows = []


for exercise in sorted(
    df[
        "exercise"
    ].unique()
):

    comparison_rows.append(
        {
            "exercise":
                exercise,

            "pose_absolute_dz":
                dz_wide.loc[
                    exercise,
                    "Pose",
                ],

            "wearable_absolute_dz":
                dz_wide.loc[
                    exercise,
                    "Wearable",
                ],

            "hr_absolute_dz":
                dz_wide.loc[
                    exercise,
                    "Physiological",
                ],

            "pose_endpoint_consistency_percent":
                consistency_wide.loc[
                    exercise,
                    "Pose",
                ],

            "wearable_endpoint_consistency_percent":
                consistency_wide.loc[
                    exercise,
                    "Wearable",
                ],

            "hr_endpoint_consistency_percent":
                consistency_wide.loc[
                    exercise,
                    "Physiological",
                ],

            "pose_monotonic_consistency_percent":
                monotonic_wide.loc[
                    exercise,
                    "Pose",
                ],

            "wearable_monotonic_consistency_percent":
                monotonic_wide.loc[
                    exercise,
                    "Wearable",
                ],

            "hr_monotonic_consistency_percent":
                monotonic_wide.loc[
                    exercise,
                    "Physiological",
                ],
        }
    )


matched_comparison = pd.DataFrame(
    comparison_rows
)


matched_comparison.to_csv(
    TABLES_OUT
    / "06_matched_three_modality_comparison.csv",
    index=False,
)


# ============================================================
# OVERALL DESCRIPTIVE SUMMARY
# ============================================================

# IMPORTANT:
#
# This is calculated from the 9 exercise-level modality summaries.
#
# We do NOT pool all feature rows together because Pose/Watch each
# contain four features while HR contains one.
#
# Therefore each exercise contributes one modality summary.

overall_summary = (
    exercise_modality_summary
    .groupby(
        "modality",
        as_index=False,
    )
    .agg(

        n_exercises=(
            "exercise",
            "nunique",
        ),

        median_exercise_absolute_dz=(
            "median_absolute_dz",
            "median",
        ),

        mean_exercise_absolute_dz=(
            "median_absolute_dz",
            "mean",
        ),

        median_exercise_endpoint_consistency_percent=(
            "median_endpoint_consistency_percent",
            "median",
        ),

        mean_exercise_endpoint_consistency_percent=(
            "median_endpoint_consistency_percent",
            "mean",
        ),

        median_exercise_monotonic_consistency_percent=(
            "median_monotonic_consistency_percent",
            "median",
        ),

        mean_exercise_monotonic_consistency_percent=(
            "median_monotonic_consistency_percent",
            "mean",
        ),
    )
)


overall_summary.to_csv(
    TABLES_OUT
    / "06_matched_overall_modality_summary.csv",
    index=False,
)


# ============================================================
# FIGURE 1 — MATCHED ABSOLUTE dz
# ============================================================

plot_dz = (
    exercise_modality_summary
    .pivot(
        index="exercise",
        columns="modality",
        values="median_absolute_dz",
    )
    .sort_index()
)


x = np.arange(
    len(
        plot_dz.index
    )
)


width = 0.25


fig, ax = plt.subplots(
    figsize=(
        14,
        7,
    )
)


ax.bar(
    x - width,
    plot_dz[
        "Pose"
    ],
    width,
    label="Pose",
)


ax.bar(
    x,
    plot_dz[
        "Wearable"
    ],
    width,
    label="Wearable",
)


ax.bar(
    x + width,
    plot_dz[
        "Physiological"
    ],
    width,
    label="Heart rate",
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        exercise.replace(
            "_",
            " ",
        )
        for exercise in plot_dz.index
    ],
    rotation=35,
    ha="right",
)


ax.set_ylabel(
    "Absolute paired Cohen's dz"
)


ax.set_xlabel(
    "Exercise"
)


ax.set_title(
    "Matched-Cohort Repeated-Performance Responsiveness"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "06_matched_three_modality_absolute_dz.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# FIGURE 2 — MATCHED DIRECTIONAL CONSISTENCY
# ============================================================

plot_consistency = (
    exercise_modality_summary
    .pivot(
        index="exercise",
        columns="modality",
        values="median_endpoint_consistency_percent",
    )
    .reindex(
        plot_dz.index
    )
)


fig, ax = plt.subplots(
    figsize=(
        14,
        7,
    )
)


ax.bar(
    x - width,
    plot_consistency[
        "Pose"
    ],
    width,
    label="Pose",
)


ax.bar(
    x,
    plot_consistency[
        "Wearable"
    ],
    width,
    label="Wearable",
)


ax.bar(
    x + width,
    plot_consistency[
        "Physiological"
    ],
    width,
    label="Heart rate",
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        exercise.replace(
            "_",
            " ",
        )
        for exercise in plot_consistency.index
    ],
    rotation=35,
    ha="right",
)


ax.set_ylabel(
    "Endpoint directional consistency (%)"
)


ax.set_xlabel(
    "Exercise"
)


ax.set_ylim(
    0,
    100,
)


ax.set_title(
    "Matched-Cohort Directional Consistency by Modality"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "06_matched_three_modality_consistency.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# FIGURE 3 — MAGNITUDE VS CONSISTENCY
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        11,
        8,
    )
)


for modality in [
    "Pose",
    "Wearable",
    "Physiological",
]:

    modality_df = (
        exercise_modality_summary.loc[
            exercise_modality_summary[
                "modality"
            ] == modality
        ]
    )


    label = (
        "Heart rate"
        if modality == "Physiological"
        else modality
    )


    ax.scatter(
        modality_df[
            "median_endpoint_consistency_percent"
        ],
        modality_df[
            "median_absolute_dz"
        ],
        s=75,
        label=label,
    )


ax.set_xlabel(
    "Endpoint directional consistency (%)"
)


ax.set_ylabel(
    "Absolute paired Cohen's dz"
)


ax.set_title(
    "Matched Modality Responsiveness: Magnitude vs Consistency"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "06_matched_modality_magnitude_vs_consistency.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# METHOD RECORD
# ============================================================

method_record = """
SCRIPT 06 — MATCHED POSE / WEARABLE / HEART-RATE COMPARISON

Purpose
-------
Provide a fair descriptive comparison of the three sensing modalities.

Reason for matched cohort
-------------------------
Pose and wearable analyses originally contained 181 complete trajectories.
Heart rate contained only 154 complete trajectories.

All three modalities were therefore recalculated using only the same
154 HR-complete trajectories.

Matched cohort
--------------
462 rows
154 complete A1/A2/A3 workout/exercise trajectories
9 true subjects
9 exercises

Feature groups
--------------
Pose:
4 frozen whole-body pose summary features.

Wearable:
4 frozen bilateral smartwatch inertial summary features.

Physiological:
1 combined mean heart-rate feature.

Metrics
-------
Feature-level:
- A1/A2/A3 values
- paired A3-A1 change
- percentage change
- endpoint directional consistency
- monotonic A1-A2-A3 consistency
- paired Cohen's dz

Exercise-level modality summary
-------------------------------
Pose and Wearable:
median metric across the four frozen features.

Heart Rate:
the single HR feature itself.

Interpretation limitation
-------------------------
Heart Rate consists of one feature whereas Pose and Wearable consist
of four features each.

Therefore the numerical modality summaries do NOT constitute proof
that one modality is superior to another.

They describe responsiveness of the selected measurements to repeated
performance on the same trajectories.

No combined sensitivity score was created.

No significance threshold was introduced.

No fatigue or behavioural-state inference was performed.

Fusion
------
Fusion is NOT performed in Script 06.

Later fusion analysis will combine the ORIGINAL attempt-level sensor
features and test whether multimodal input improves discrimination of
A1 versus A3 under true-subject held-out validation.

Cohen's dz values are never fused.
"""


(
    LOGS_OUT
    / "06_method_record.txt"
).write_text(
    method_record.strip(),
    encoding="utf-8",
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n" + "-" * 84)
print("MATCHED THREE-MODALITY COMPARISON")
print("-" * 84)


print(
    matched_comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 84)
print("OVERALL MATCHED DESCRIPTIVE SUMMARY")
print("-" * 84)


print(
    overall_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 84)
print("MATCHED FEATURE-LEVEL RESULTS — LARGEST |dz|")
print("-" * 84)


display = (
    feature_summary
    .sort_values(
        "absolute_dz",
        ascending=False,
    )
)


print(
    display[
        [
            "exercise",
            "feature",
            "modality",
            "median_percent_change_A1_A3",
            "endpoint_direction_consistency_percent",
            "monotonic_A1_A2_A3_consistency_percent",
            "paired_cohens_dz",
        ]
    ]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "=" * 84)
print("SCRIPT 06 COMPLETE")
print("=" * 84)


print(
    "\nAll modalities were recalculated on the same "
    "154 complete trajectories."
)


print(
    "\nNo modality has been labelled as universally better."
)


print(
    "\nNext stage: multimodal fusion using the original "
    "attempt-level features."
)