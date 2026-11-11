"""
04_compare_pose_watch_modalities.py

PRIMARY THESIS ANALYSIS — MODALITY SENSITIVITY COMPARISON

Purpose
-------
Compare how strongly the frozen pose-derived and wearable-derived
feature groups respond to repeated exercise attempts.

This script uses the results already calculated in Script 02.

It does NOT:
- create new sensor features,
- change the A1/A2/A3 cohort,
- perform significance testing,
- infer fatigue,
- infer behavioural state,
- create an arbitrary combined "sensitivity score",
- include heart rate.

Instead, Pose and Wearable are compared descriptively using:

1. Absolute paired Cohen's dz
2. Endpoint directional consistency
3. Monotonic A1 -> A2 -> A3 consistency
4. Absolute median percentage change

The primary modality comparison focuses mainly on:
- median absolute dz across the four features in each modality
- median directional consistency across the four features

Percentage change is retained as supporting information but is NOT
used alone to determine modality sensitivity because feature scales
and definitions differ substantially.

IMPORTANT
---------
The 36 feature/exercise comparisons within each modality are not
independent statistical observations.

The overall modality summaries produced here are therefore
DESCRIPTIVE and should not be interpreted as inferential tests.
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
    / "tables"
    / "02_pose_watch_feature_change_summary.csv"
)

TABLES_OUT = RESULTS_ROOT / "tables"
FIGURES_OUT = RESULTS_ROOT / "figures"
LOGS_OUT = RESULTS_ROOT / "logs"

for folder in [
    TABLES_OUT,
    FIGURES_OUT,
    LOGS_OUT,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        "\nScript 02 pose/watch summary could not be found.\n\n"
        f"Expected:\n{INPUT_PATH}\n"
    )


df = pd.read_csv(
    INPUT_PATH
)


print("\n" + "=" * 82)
print("MM-FIT REPEATED PERFORMANCE ANALYSIS")
print("SCRIPT 04 — POSE VS WEARABLE MODALITY COMPARISON")
print("=" * 82)

print(
    f"\nInput:\n{INPUT_PATH}"
)

print(
    f"\nRows loaded: {len(df)}"
)

print(
    f"Exercises: {df['exercise'].nunique()}"
)

print(
    f"Modalities: {', '.join(sorted(df['modality'].unique()))}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "exercise",
    "feature",
    "modality",
    "n_trajectories",
    "n_true_subjects",
    "median_percent_change_A1_A3",
    "endpoint_direction_consistency_percent",
    "monotonic_A1_A2_A3_consistency_percent",
    "paired_cohens_dz",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# Only Pose and Wearable should be present here.

unexpected_modalities = set(
    df["modality"].unique()
) - {
    "Pose",
    "Wearable",
}


if unexpected_modalities:

    raise ValueError(
        "\nUnexpected modality values found:\n"
        + "\n".join(
            sorted(
                unexpected_modalities
            )
        )
    )


# ============================================================
# CHECK FOUR FROZEN FEATURES PER MODALITY
# ============================================================

feature_counts = (
    df
    .groupby(
        [
            "exercise",
            "modality",
        ]
    )["feature"]
    .nunique()
)


bad_feature_counts = feature_counts[
    feature_counts != 4
]


if len(
    bad_feature_counts
) > 0:

    print(
        "\nWARNING:"
        "\nSome exercise/modality combinations do not contain "
        "exactly four features."
    )

    print(
        bad_feature_counts
    )


# ============================================================
# DERIVED DESCRIPTIVE VARIABLES
# ============================================================

analysis = df.copy()


# Absolute effect size:
# direction is preserved in the original paired_cohens_dz column,
# while absolute value represents magnitude.

analysis["absolute_dz"] = (
    analysis[
        "paired_cohens_dz"
    ].abs()
)


# Absolute percentage change is supporting information only.

analysis[
    "absolute_median_percent_change"
] = (
    analysis[
        "median_percent_change_A1_A3"
    ].abs()
)


# Save an augmented feature-level table.

analysis.to_csv(
    TABLES_OUT
    / "04_pose_watch_feature_level_comparison.csv",
    index=False,
)


# ============================================================
# EXERCISE x MODALITY SUMMARY
# ============================================================

exercise_modality = (
    analysis
    .groupby(
        [
            "exercise",
            "modality",
        ],
        as_index=False,
    )
    .agg(

        n_features=(
            "feature",
            "nunique",
        ),

        n_trajectories=(
            "n_trajectories",
            "first",
        ),

        n_true_subjects=(
            "n_true_subjects",
            "first",
        ),

        median_absolute_dz=(
            "absolute_dz",
            "median",
        ),

        mean_absolute_dz=(
            "absolute_dz",
            "mean",
        ),

        maximum_absolute_dz=(
            "absolute_dz",
            "max",
        ),

        median_endpoint_consistency_percent=(
            "endpoint_direction_consistency_percent",
            "median",
        ),

        mean_endpoint_consistency_percent=(
            "endpoint_direction_consistency_percent",
            "mean",
        ),

        median_monotonic_consistency_percent=(
            "monotonic_A1_A2_A3_consistency_percent",
            "median",
        ),

        median_absolute_percent_change=(
            "absolute_median_percent_change",
            "median",
        ),
    )
)


exercise_modality.to_csv(
    TABLES_OUT
    / "04_exercise_modality_summary.csv",
    index=False,
)


# ============================================================
# IDENTIFY MOST RESPONSIVE FEATURE WITHIN EACH MODALITY
# ============================================================

# This does NOT select features for future modelling.
# It simply documents which of the four frozen features had the
# largest absolute dz within each exercise/modality combination.

top_feature_rows = []


for (
    exercise,
    modality
), group in analysis.groupby(
    [
        "exercise",
        "modality",
    ]
):

    group = group.sort_values(
        "absolute_dz",
        ascending=False,
    )

    top = group.iloc[0]

    top_feature_rows.append(
        {
            "exercise":
                exercise,

            "modality":
                modality,

            "feature_with_largest_absolute_dz":
                top["feature"],

            "paired_cohens_dz":
                top["paired_cohens_dz"],

            "absolute_dz":
                top["absolute_dz"],

            "median_percent_change_A1_A3":
                top[
                    "median_percent_change_A1_A3"
                ],

            "endpoint_direction_consistency_percent":
                top[
                    "endpoint_direction_consistency_percent"
                ],

            "monotonic_A1_A2_A3_consistency_percent":
                top[
                    "monotonic_A1_A2_A3_consistency_percent"
                ],
        }
    )


top_features = pd.DataFrame(
    top_feature_rows
)


top_features.to_csv(
    TABLES_OUT
    / "04_top_feature_by_exercise_and_modality.csv",
    index=False,
)


# ============================================================
# DIRECT POSE VS WEARABLE EXERCISE COMPARISON
# ============================================================

dz_wide = (
    exercise_modality
    .pivot(
        index="exercise",
        columns="modality",
        values="median_absolute_dz",
    )
)


consistency_wide = (
    exercise_modality
    .pivot(
        index="exercise",
        columns="modality",
        values="median_endpoint_consistency_percent",
    )
)


monotonic_wide = (
    exercise_modality
    .pivot(
        index="exercise",
        columns="modality",
        values="median_monotonic_consistency_percent",
    )
)


comparison_rows = []


for exercise in sorted(
    analysis["exercise"].unique()
):

    pose_dz = dz_wide.loc[
        exercise,
        "Pose",
    ]

    wearable_dz = dz_wide.loc[
        exercise,
        "Wearable",
    ]

    pose_consistency = (
        consistency_wide.loc[
            exercise,
            "Pose",
        ]
    )

    wearable_consistency = (
        consistency_wide.loc[
            exercise,
            "Wearable",
        ]
    )

    pose_monotonic = (
        monotonic_wide.loc[
            exercise,
            "Pose",
        ]
    )

    wearable_monotonic = (
        monotonic_wide.loc[
            exercise,
            "Wearable",
        ]
    )


    # These labels simply state which modality has the numerically
    # larger descriptive value. They are not significance results.

    if wearable_dz > pose_dz:

        higher_dz = "Wearable"

    elif pose_dz > wearable_dz:

        higher_dz = "Pose"

    else:

        higher_dz = "Equal"


    if wearable_consistency > pose_consistency:

        higher_consistency = "Wearable"

    elif pose_consistency > wearable_consistency:

        higher_consistency = "Pose"

    else:

        higher_consistency = "Equal"


    comparison_rows.append(
        {
            "exercise":
                exercise,

            "pose_median_absolute_dz":
                pose_dz,

            "wearable_median_absolute_dz":
                wearable_dz,

            "wearable_minus_pose_absolute_dz":
                wearable_dz
                - pose_dz,

            "higher_median_absolute_dz":
                higher_dz,

            "pose_median_endpoint_consistency_percent":
                pose_consistency,

            "wearable_median_endpoint_consistency_percent":
                wearable_consistency,

            "wearable_minus_pose_endpoint_consistency":
                wearable_consistency
                - pose_consistency,

            "higher_endpoint_consistency":
                higher_consistency,

            "pose_median_monotonic_consistency_percent":
                pose_monotonic,

            "wearable_median_monotonic_consistency_percent":
                wearable_monotonic,
        }
    )


pose_vs_watch = pd.DataFrame(
    comparison_rows
)


pose_vs_watch.to_csv(
    TABLES_OUT
    / "04_pose_vs_wearable_by_exercise.csv",
    index=False,
)


# ============================================================
# OVERALL DESCRIPTIVE MODALITY SUMMARY
# ============================================================

overall_modality = (
    analysis
    .groupby(
        "modality",
        as_index=False,
    )
    .agg(

        n_feature_exercise_comparisons=(
            "feature",
            "size",
        ),

        n_exercises=(
            "exercise",
            "nunique",
        ),

        median_absolute_dz=(
            "absolute_dz",
            "median",
        ),

        mean_absolute_dz=(
            "absolute_dz",
            "mean",
        ),

        median_endpoint_consistency_percent=(
            "endpoint_direction_consistency_percent",
            "median",
        ),

        mean_endpoint_consistency_percent=(
            "endpoint_direction_consistency_percent",
            "mean",
        ),

        median_monotonic_consistency_percent=(
            "monotonic_A1_A2_A3_consistency_percent",
            "median",
        ),

        mean_monotonic_consistency_percent=(
            "monotonic_A1_A2_A3_consistency_percent",
            "mean",
        ),

        median_absolute_percent_change=(
            "absolute_median_percent_change",
            "median",
        ),
    )
)


overall_modality.to_csv(
    TABLES_OUT
    / "04_overall_modality_descriptive_summary.csv",
    index=False,
)


# ============================================================
# FIGURE 1
# MEDIAN ABSOLUTE dz BY EXERCISE
# ============================================================

plot_dz = (
    exercise_modality
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

width = 0.36


fig, ax = plt.subplots(
    figsize=(
        13,
        7,
    )
)


ax.bar(
    x - width / 2,
    plot_dz["Pose"],
    width,
    label="Pose",
)


ax.bar(
    x + width / 2,
    plot_dz["Wearable"],
    width,
    label="Wearable",
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
    "Median absolute paired Cohen's dz"
)


ax.set_xlabel(
    "Exercise"
)


ax.set_title(
    "Pose vs Wearable: Standardised Repeated-Performance Change"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "04_pose_vs_wearable_absolute_dz.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# FIGURE 2
# DIRECTIONAL CONSISTENCY BY EXERCISE
# ============================================================

plot_consistency = (
    exercise_modality
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
        13,
        7,
    )
)


ax.bar(
    x - width / 2,
    plot_consistency["Pose"],
    width,
    label="Pose",
)


ax.bar(
    x + width / 2,
    plot_consistency["Wearable"],
    width,
    label="Wearable",
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
    "Median endpoint directional consistency (%)"
)


ax.set_xlabel(
    "Exercise"
)


ax.set_ylim(
    0,
    100,
)


ax.set_title(
    "Pose vs Wearable: Directional Consistency Across Repeated Attempts"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "04_pose_vs_wearable_direction_consistency.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# FIGURE 3
# EFFECT SIZE + CONSISTENCY TOGETHER
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        8,
    )
)


for modality in [
    "Pose",
    "Wearable",
]:

    modality_df = (
        exercise_modality.loc[
            exercise_modality[
                "modality"
            ] == modality
        ]
    )


    ax.scatter(
        modality_df[
            "median_endpoint_consistency_percent"
        ],
        modality_df[
            "median_absolute_dz"
        ],
        label=modality,
        s=70,
    )


    for _, row in modality_df.iterrows():

        label = (
            row["exercise"]
            .replace(
                "dumbbell_",
                "",
            )
            .replace(
                "_",
                " ",
            )
        )


        ax.annotate(
            label,
            (
                row[
                    "median_endpoint_consistency_percent"
                ],
                row[
                    "median_absolute_dz"
                ],
            ),
            xytext=(
                4,
                4,
            ),
            textcoords="offset points",
            fontsize=8,
        )


ax.set_xlabel(
    "Median endpoint directional consistency (%)"
)


ax.set_ylabel(
    "Median absolute paired Cohen's dz"
)


ax.set_title(
    "Exercise-Level Modality Sensitivity: Magnitude vs Consistency"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "04_modality_magnitude_vs_consistency.png",
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
SCRIPT 04 — POSE VS WEARABLE MODALITY COMPARISON

Purpose:
Compare the sensitivity of the frozen pose and wearable feature groups
to repeated-performance change across exercises.

Input:
02_pose_watch_feature_change_summary.csv

Feature groups:
Pose = 4 frozen whole-body pose summary features.
Wearable = 4 frozen bilateral smartwatch inertial summary features.

No new features were introduced.

Primary descriptive modality metrics:
1. Median absolute paired Cohen's dz across the four features.
2. Median endpoint directional consistency across the four features.
3. Median monotonic A1-A2-A3 consistency across the four features.

Supporting metric:
Median absolute percentage change.

Percentage change is not used alone for modality comparison because
feature scale and construction differ between pose and wearable signals.

No combined sensitivity score was created.

No statistical significance test was performed.

No threshold such as dz > 0.5 or consistency > 70% was introduced
to classify features as successful or unsuccessful.

Heart rate was intentionally excluded and will be analysed separately.

Interpretation limitation:
The feature/exercise comparisons are not independent observations.
Overall modality summaries are descriptive and must not be interpreted
as inferential evidence that one modality is statistically superior.

The purpose is to identify whether sensitivity differs by exercise and
whether Pose and Wearable appear to capture complementary patterns.
"""


(
    LOGS_OUT
    / "04_method_record.txt"
).write_text(
    method_record.strip(),
    encoding="utf-8",
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n" + "-" * 82)
print("OVERALL DESCRIPTIVE MODALITY SUMMARY")
print("-" * 82)


print(
    overall_modality.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 82)
print("EXERCISE-LEVEL POSE VS WEARABLE COMPARISON")
print("-" * 82)


display_columns = [
    "exercise",

    "pose_median_absolute_dz",
    "wearable_median_absolute_dz",
    "higher_median_absolute_dz",

    "pose_median_endpoint_consistency_percent",
    "wearable_median_endpoint_consistency_percent",
    "higher_endpoint_consistency",
]


print(
    pose_vs_watch[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 82)
print("LARGEST-|dz| FEATURE WITHIN EACH EXERCISE/MODALITY")
print("-" * 82)


print(
    top_features.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "=" * 82)
print("SCRIPT 04 COMPLETE")
print("=" * 82)


print(
    "\nTables saved to:"
)

print(
    TABLES_OUT
)


print(
    "\nFigures saved to:"
)

print(
    FIGURES_OUT
)


print(
    "\nHeart rate was NOT included."
)


print(
    "\nNo arbitrary combined sensitivity score or significance "
    "threshold was introduced."
)