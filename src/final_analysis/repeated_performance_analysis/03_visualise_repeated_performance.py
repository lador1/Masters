"""
03_visualise_repeated_performance.py

PRIMARY THESIS ANALYSIS — VISUALISATION AND QC

Purpose
-------
Visualise the pose and smartwatch repeated-performance results from
Scripts 01-02 without cherry-picking individual findings.

This script:
1. Checks whether exercise duration or repetition count changes across A1/A2/A3.
2. Creates complete heatmaps for ALL frozen pose/watch features.
3. Creates paired-normalised A1->A2->A3 progression plots for ALL 8
   pose/watch features.
4. Saves a figure manifest and interpretation notes.

Heart rate is intentionally kept separate at this stage.
No HR figures are mixed into the pose/watch figures.

Important interpretation rule
-----------------------------
These figures describe repeated-performance change.
They do NOT independently indicate fatigue, engagement, frustration,
clinical deterioration, or movement quality.

watch_acc_mag_mean caution
--------------------------
The observed mean resultant acceleration values are approximately 9.8-10 m/s^2,
which strongly suggests that the gravitational component remains present.
Therefore watch_acc_mag_mean must NOT be described as pure dynamic movement
intensity. watch_acc_mag_std is a cleaner descriptor of dynamic variability.
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

SUMMARY_INPUT = (
    RESULTS_ROOT
    / "tables"
    / "02_pose_watch_feature_change_summary.csv"
)

TABLES_OUT = RESULTS_ROOT / "tables"
FIGURES_OUT = RESULTS_ROOT / "figures"
LOGS_OUT = RESULTS_ROOT / "logs"

for folder in [TABLES_OUT, FIGURES_OUT, LOGS_OUT]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# FROZEN FEATURE SET
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

ALL_FEATURES = POSE_FEATURES + WATCH_FEATURES

FEATURE_LABELS = {
    "pose_movement_mean": "Pose movement mean",
    "pose_movement_std": "Pose movement variability",
    "pose_movement_max": "Pose movement maximum",
    "pose_range_mean": "Pose range mean",
    "watch_acc_mag_mean": "Watch acceleration magnitude mean",
    "watch_acc_mag_std": "Watch acceleration magnitude variability",
    "watch_gyr_mag_mean": "Watch gyroscope magnitude mean",
    "watch_gyr_mag_std": "Watch gyroscope magnitude variability",
}


# ============================================================
# LOAD
# ============================================================

if not PRIMARY_INPUT.exists():
    raise FileNotFoundError(
        f"\nMissing Script 01 dataset:\n{PRIMARY_INPUT}"
    )

if not SUMMARY_INPUT.exists():
    raise FileNotFoundError(
        f"\nMissing Script 02 pose/watch summary:\n{SUMMARY_INPUT}"
    )

df = pd.read_csv(PRIMARY_INPUT)
summary = pd.read_csv(SUMMARY_INPUT)


print("\n" + "=" * 80)
print("MM-FIT REPEATED PERFORMANCE ANALYSIS")
print("SCRIPT 03 — VISUALISATION AND QUALITY CONTROL")
print("=" * 80)

print(f"\nPrimary dataset rows: {len(df)}")
print(
    "Complete trajectories: "
    f"{df.groupby(['workout_session', 'exercise']).ngroups}"
)
print(f"Exercises: {df['exercise'].nunique()}")


# ============================================================
# QC 1 — DURATION AND REPETITION COUNT
# ============================================================

qc_rows = []

for (exercise, attempt), group in df.groupby(
    ["exercise", "attempt"]
):
    qc_rows.append(
        {
            "exercise": exercise,
            "attempt": attempt,
            "n_segments": len(group),

            "duration_median": group["duration"].median(),
            "duration_Q1": group["duration"].quantile(0.25),
            "duration_Q3": group["duration"].quantile(0.75),

            "reps_median": group["reps"].median(),
            "reps_Q1": group["reps"].quantile(0.25),
            "reps_Q3": group["reps"].quantile(0.75),
        }
    )

qc_summary = pd.DataFrame(qc_rows)

qc_summary["attempt_number"] = (
    qc_summary["attempt"]
    .str.replace("A", "", regex=False)
    .astype(int)
)

qc_summary = qc_summary.sort_values(
    ["exercise", "attempt_number"]
)

qc_summary.to_csv(
    TABLES_OUT / "03_duration_rep_qc_summary.csv",
    index=False,
)


# ============================================================
# QC FIGURE — DURATION
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

for exercise, group in qc_summary.groupby("exercise"):
    group = group.sort_values("attempt_number")

    ax.plot(
        group["attempt_number"],
        group["duration_median"],
        marker="o",
        label=exercise,
    )

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["A1", "A2", "A3"])
ax.set_xlabel("Repeated attempt")
ax.set_ylabel("Median segment duration")
ax.set_title(
    "QC: Exercise Segment Duration Across Repeated Attempts"
)
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
fig.tight_layout()

duration_path = (
    FIGURES_OUT
    / "03_QC_duration_progression.png"
)

fig.savefig(
    duration_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# QC FIGURE — REPETITION COUNT
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

for exercise, group in qc_summary.groupby("exercise"):
    group = group.sort_values("attempt_number")

    ax.plot(
        group["attempt_number"],
        group["reps_median"],
        marker="o",
        label=exercise,
    )

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["A1", "A2", "A3"])
ax.set_xlabel("Repeated attempt")
ax.set_ylabel("Median labelled repetition count")
ax.set_title(
    "QC: Labelled Repetition Count Across Repeated Attempts"
)
ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
fig.tight_layout()

reps_path = (
    FIGURES_OUT
    / "03_QC_repetition_progression.png"
)

fig.savefig(
    reps_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ============================================================
# HEATMAP HELPER
# ============================================================

def save_heatmap(
    data,
    features,
    value_column,
    title,
    filename,
    number_format=".2f",
):
    matrix = (
        data[
            data["feature"].isin(features)
        ]
        .pivot(
            index="exercise",
            columns="feature",
            values=value_column,
        )
    )

    matrix = matrix.reindex(
        columns=features
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    values = matrix.to_numpy(dtype=float)

    if value_column == "paired_cohens_dz":
        max_abs = np.nanmax(np.abs(values))

        image = ax.imshow(
            values,
            aspect="auto",
            vmin=-max_abs,
            vmax=max_abs,
        )

    else:
        image = ax.imshow(
            values,
            aspect="auto",
        )

    ax.set_xticks(
        range(len(matrix.columns))
    )

    ax.set_xticklabels(
        [
            FEATURE_LABELS.get(c, c)
            for c in matrix.columns
        ],
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        range(len(matrix.index))
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_title(title)

    for row_idx in range(
        len(matrix.index)
    ):
        for col_idx in range(
            len(matrix.columns)
        ):
            value = values[
                row_idx,
                col_idx
            ]

            if not np.isnan(value):
                ax.text(
                    col_idx,
                    row_idx,
                    format(
                        value,
                        number_format,
                    ),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.colorbar(
        image,
        ax=ax,
        shrink=0.85,
    )

    fig.tight_layout()

    output_path = (
        FIGURES_OUT
        / filename
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# ============================================================
# COMPLETE EFFECT-SIZE HEATMAPS
# ============================================================

pose_dz_path = save_heatmap(
    summary,
    POSE_FEATURES,
    "paired_cohens_dz",
    "Pose Features: Paired A1-to-A3 Standardised Change (Cohen's dz)",
    "03_pose_effect_size_heatmap.png",
)

watch_dz_path = save_heatmap(
    summary,
    WATCH_FEATURES,
    "paired_cohens_dz",
    "Wearable Features: Paired A1-to-A3 Standardised Change (Cohen's dz)",
    "03_watch_effect_size_heatmap.png",
)


# ============================================================
# COMPLETE CONSISTENCY HEATMAPS
# ============================================================

pose_consistency_path = save_heatmap(
    summary,
    POSE_FEATURES,
    "endpoint_direction_consistency_percent",
    "Pose Features: A1-to-A3 Directional Consistency (%)",
    "03_pose_direction_consistency_heatmap.png",
    number_format=".0f",
)

watch_consistency_path = save_heatmap(
    summary,
    WATCH_FEATURES,
    "endpoint_direction_consistency_percent",
    "Wearable Features: A1-to-A3 Directional Consistency (%)",
    "03_watch_direction_consistency_heatmap.png",
    number_format=".0f",
)


# ============================================================
# PAIRED NORMALISED A1-A2-A3 PROGRESSION
# ============================================================

progression_rows = []


for exercise in sorted(df["exercise"].unique()):

    exercise_df = df.loc[
        df["exercise"] == exercise
    ].copy()

    for feature in ALL_FEATURES:

        wide = exercise_df.pivot_table(
            index=[
                "workout_session",
                "true_subject",
            ],
            columns="attempt",
            values=feature,
            aggfunc="first",
        )

        if not all(
            attempt in wide.columns
            for attempt in ["A1", "A2", "A3"]
        ):
            continue

        wide = wide[
            ["A1", "A2", "A3"]
        ].dropna()

        wide = wide.loc[
            ~np.isclose(wide["A1"], 0)
        ].copy()

        for attempt in [
            "A1",
            "A2",
            "A3",
        ]:

            percent_change = (
                (
                    wide[attempt]
                    - wide["A1"]
                )
                / wide["A1"]
                * 100.0
            )

            progression_rows.append(
                {
                    "exercise": exercise,
                    "feature": feature,
                    "attempt": attempt,
                    "attempt_number": int(
                        attempt.replace(
                            "A",
                            "",
                        )
                    ),
                    "n_trajectories":
                        len(percent_change),

                    "median_percent_change_from_A1":
                        percent_change.median(),

                    "Q1_percent_change_from_A1":
                        percent_change.quantile(0.25),

                    "Q3_percent_change_from_A1":
                        percent_change.quantile(0.75),
                }
            )


progression = pd.DataFrame(
    progression_rows
)

progression.to_csv(
    TABLES_OUT
    / "03_paired_normalised_progression.csv",
    index=False,
)


# ============================================================
# ONE COMPLETE FIGURE PER FEATURE
# ============================================================

progression_paths = []


for feature in ALL_FEATURES:

    feature_df = progression.loc[
        progression["feature"] == feature
    ].copy()

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for exercise, group in feature_df.groupby(
        "exercise"
    ):

        group = group.sort_values(
            "attempt_number"
        )

        ax.plot(
            group["attempt_number"],
            group[
                "median_percent_change_from_A1"
            ],
            marker="o",
            label=exercise,
        )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(
        [1, 2, 3]
    )

    ax.set_xticklabels(
        ["A1", "A2", "A3"]
    )

    ax.set_xlabel(
        "Repeated attempt"
    )

    ax.set_ylabel(
        "Median within-trajectory change from A1 (%)"
    )

    ax.set_title(
        FEATURE_LABELS[feature]
        + ": Paired Repeated-Attempt Progression"
    )

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    output_path = (
        FIGURES_OUT
        / f"03_progression_{feature}.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    progression_paths.append(
        output_path
    )


# ============================================================
# INTERPRETATION NOTES
# ============================================================

notes = pd.DataFrame(
    [
        {
            "item": "watch_acc_mag_mean",
            "note":
                "Mean resultant acceleration is approximately 9.8-10 m/s^2 "
                "across exercises. This strongly suggests the static "
                "gravitational component remains present. Retain the feature "
                "for reproducibility, but do not describe it as pure dynamic "
                "movement intensity. Interpret absolute changes cautiously."
        },

        {
            "item": "watch_acc_mag_std",
            "note":
                "Variability of resultant acceleration is less dominated by "
                "the constant gravitational component and is therefore a "
                "more direct descriptor of changes in dynamic wrist motion."
        },

        {
            "item": "paired_cohens_dz",
            "note":
                "A large absolute dz can occur even when the absolute or "
                "percentage change is small if within-trajectory changes are "
                "highly consistent. Effect size must therefore be interpreted "
                "with change magnitude and directional consistency."
        },

        {
            "item": "pose_range_mean",
            "note":
                "Large changes in pose range require comparison with segment "
                "duration and labelled repetition count before biomechanical "
                "interpretation, because segment composition may influence "
                "range-based summaries."
        },

        {
            "item": "heart_rate",
            "note":
                "Heart rate remains excluded from Script 03 movement figures "
                "and is retained as a separate physiological analysis."
        },
    ]
)

notes.to_csv(
    TABLES_OUT
    / "03_interpretation_notes.csv",
    index=False,
)


# ============================================================
# FIGURE MANIFEST
# ============================================================

manifest_rows = [
    {
        "figure": duration_path.name,
        "purpose":
            "QC check for segment-duration differences across A1/A2/A3."
    },
    {
        "figure": reps_path.name,
        "purpose":
            "QC check for labelled repetition-count differences across A1/A2/A3."
    },
    {
        "figure": pose_dz_path.name,
        "purpose":
            "Complete cross-exercise pose effect-size overview."
    },
    {
        "figure": watch_dz_path.name,
        "purpose":
            "Complete cross-exercise wearable effect-size overview."
    },
    {
        "figure": pose_consistency_path.name,
        "purpose":
            "Complete cross-exercise pose directional-consistency overview."
    },
    {
        "figure": watch_consistency_path.name,
        "purpose":
            "Complete cross-exercise wearable directional-consistency overview."
    },
]

for path in progression_paths:

    feature = path.stem.replace(
        "03_progression_",
        "",
    )

    manifest_rows.append(
        {
            "figure": path.name,
            "purpose":
                "Paired A1-normalised progression across all exercises for "
                + feature
                + "."
        }
    )

manifest = pd.DataFrame(
    manifest_rows
)

manifest.to_csv(
    TABLES_OUT
    / "03_figure_manifest.csv",
    index=False,
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "-" * 80)
print("DURATION / REPETITION QC TABLE")
print("-" * 80)

qc_display = qc_summary[
    [
        "exercise",
        "attempt",
        "duration_median",
        "reps_median",
    ]
]

print(
    qc_display.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 80)
print("POSE/WATCH RESULT CHECK — LARGEST |dz| VALUES")
print("-" * 80)

display = summary.copy()

display["absolute_dz"] = (
    display["paired_cohens_dz"].abs()
)

display = display.sort_values(
    "absolute_dz",
    ascending=False,
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
    .head(15)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "=" * 80)
print("SCRIPT 03 COMPLETE")
print("=" * 80)

print(
    f"\nFigures saved to:\n{FIGURES_OUT}"
)

print(
    f"\nQC / progression tables saved to:\n{TABLES_OUT}"
)

print(
    "\nHeart rate was intentionally kept separate from these "
    "pose/watch visualisations."
)