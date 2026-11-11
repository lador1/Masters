"""
02_measure_feature_change.py

PRIMARY THESIS ANALYSIS — REPEATED-PERFORMANCE CHANGE

Purpose
-------
Measure how the frozen primary sensor features change across repeated
exercise attempts A1 -> A2 -> A3.

IMPORTANT COHORT RULE
---------------------
Pose and smartwatch features are analysed using ALL complete A1/A2/A3
exercise trajectories.

Heart rate is analysed SEPARATELY using only trajectories where HR is
available at A1, A2 and A3.

This avoids discarding valid pose/watch data simply because HR is missing.

This script is descriptive. It does NOT:
- infer fatigue,
- infer engagement,
- infer behavioural state,
- perform significance testing,
- select features based on results,
- claim one modality is "best".

For each exercise x feature, the script reports:
- A1 median
- A2 median
- A3 median
- median paired A3 - A1 change
- median paired percentage change
- dominant A1 -> A3 direction
- endpoint directional consistency
- monotonic A1 -> A2 -> A3 consistency
- paired Cohen's dz as a standardized descriptive effect size

Unit of analysis
----------------
A repeated trajectory is one:
    workout_session + exercise

Each trajectory contains:
    A1, A2, A3

The same true subject may contribute more than one workout session.
Therefore these are descriptive trajectory-level summaries, not
independent-subject inferential tests.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RESULTS_ROOT = Path(
    r"C:\Masters\results\final_analysis\repeated_performance_analysis"
)

INPUT_PATH = (
    RESULTS_ROOT
    / "data"
    / "01_repeated_attempts_primary_A1_A2_A3.csv"
)

DATA_OUT = RESULTS_ROOT / "data"
TABLES_OUT = RESULTS_ROOT / "tables"
LOGS_OUT = RESULTS_ROOT / "logs"

for folder in [DATA_OUT, TABLES_OUT, LOGS_OUT]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# FROZEN PRIMARY FEATURE SET
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
# LOAD DATA
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "\nScript 01 output could not be found.\n\n"
        f"Expected location:\n{INPUT_PATH}\n"
    )

df = pd.read_csv(INPUT_PATH)


print("\n" + "=" * 80)
print("MM-FIT REPEATED PERFORMANCE ANALYSIS")
print("SCRIPT 02 — MEASURE FEATURE CHANGE")
print("=" * 80)

print(f"\nInput:\n{INPUT_PATH}")

print("\nDataset loaded:")
print(f"Rows:                         {len(df)}")
print(f"Complete trajectories:        "
      f"{df.groupby(['workout_session', 'exercise']).ngroups}")
print(f"Workout sessions represented: {df['workout_session'].nunique()}")
print(f"True subjects represented:    {df['true_subject'].nunique()}")
print(f"Exercises represented:        {df['exercise'].nunique()}")


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "workout_session",
    "true_subject",
    "exercise",
    "attempt_number",
    "attempt",
    "hr_complete_A1_A2_A3",
] + POSE_FEATURES + WATCH_FEATURES + HR_FEATURES

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nRequired columns are missing:\n"
        + "\n".join(missing_columns)
    )


# Confirm each primary trajectory contains exactly A1, A2 and A3.

attempt_sets = (
    df
    .groupby(["workout_session", "exercise"])["attempt"]
    .apply(lambda x: tuple(sorted(x.unique())))
)

bad_attempt_sets = attempt_sets[
    attempt_sets != ("A1", "A2", "A3")
]

if len(bad_attempt_sets) > 0:
    raise RuntimeError(
        "\nSome primary trajectories do not contain exactly A1, A2 and A3."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def paired_dz(differences):
    """
    Paired Cohen's dz.

    D_i = A3_i - A1_i
    dz  = mean(D) / SD(D)

    Used here only as a standardized descriptive effect size.
    """

    values = pd.Series(differences).dropna().astype(float)

    if len(values) < 2:
        return np.nan

    sd = values.std(ddof=1)

    if np.isclose(sd, 0):
        return np.nan

    return values.mean() / sd


def safe_percent_change(a1, a3):
    """
    Percentage change:
        ((A3 - A1) / A1) * 100

    Returns NaN when A1 is zero or missing.
    """

    if pd.isna(a1) or pd.isna(a3):
        return np.nan

    if np.isclose(a1, 0):
        return np.nan

    return ((a3 - a1) / a1) * 100.0


def direction_from_value(value):
    """
    Convert a signed change value into a simple direction label.
    """

    if pd.isna(value):
        return "unknown"

    if value > 0:
        return "increase"

    if value < 0:
        return "decrease"

    return "no_change"


# ============================================================
# BUILD TRAJECTORY-LEVEL CHANGE TABLE
# ============================================================

trajectory_rows = []

all_features = (
    POSE_FEATURES
    + WATCH_FEATURES
    + HR_FEATURES
)


for exercise in sorted(df["exercise"].unique()):

    exercise_df = df.loc[
        df["exercise"] == exercise
    ].copy()

    for feature in all_features:

        modality = FEATURE_MODALITY[feature]

        feature_df = exercise_df.copy()

        # HR uses only fully HR-complete A1/A2/A3 trajectories.
        if feature in HR_FEATURES:
            feature_df = feature_df.loc[
                feature_df["hr_complete_A1_A2_A3"] == True
            ].copy()

        pivot = feature_df.pivot_table(
            index=[
                "workout_session",
                "true_subject",
            ],
            columns="attempt",
            values=feature,
            aggfunc="first",
        ).reset_index()

        # Ensure expected attempt columns exist.
        for attempt in ["A1", "A2", "A3"]:
            if attempt not in pivot.columns:
                pivot[attempt] = np.nan

        # A feature-level trajectory must have all three values.
        pivot = pivot.dropna(
            subset=["A1", "A2", "A3"]
        ).copy()

        for _, row in pivot.iterrows():

            a1 = float(row["A1"])
            a2 = float(row["A2"])
            a3 = float(row["A3"])

            delta_a1_a2 = a2 - a1
            delta_a2_a3 = a3 - a2
            delta_a1_a3 = a3 - a1

            pct_a1_a3 = safe_percent_change(
                a1,
                a3,
            )

            trajectory_rows.append(
                {
                    "exercise": exercise,
                    "feature": feature,
                    "modality": modality,
                    "workout_session": row["workout_session"],
                    "true_subject": int(row["true_subject"]),

                    "A1": a1,
                    "A2": a2,
                    "A3": a3,

                    "delta_A1_A2": delta_a1_a2,
                    "delta_A2_A3": delta_a2_a3,
                    "delta_A1_A3": delta_a1_a3,

                    "percent_change_A1_A3": pct_a1_a3,

                    "endpoint_direction":
                        direction_from_value(delta_a1_a3),
                }
            )


trajectory_changes = pd.DataFrame(
    trajectory_rows
)


trajectory_changes.to_csv(
    DATA_OUT / "02_trajectory_level_feature_changes.csv",
    index=False,
)


# ============================================================
# SUMMARISE EACH EXERCISE x FEATURE
# ============================================================

summary_rows = []


for (exercise, feature), group in trajectory_changes.groupby(
    ["exercise", "feature"]
):

    group = group.copy()

    modality = group["modality"].iloc[0]

    median_delta = group["delta_A1_A3"].median()
    dominant_direction = direction_from_value(median_delta)

    # --------------------------------------------------------
    # Endpoint directional consistency
    #
    # If the median A1->A3 change is positive:
    #     proportion of trajectories where A3 > A1
    #
    # If negative:
    #     proportion where A3 < A1
    #
    # If exactly zero:
    #     proportion where A3 == A1
    # --------------------------------------------------------

    if dominant_direction == "increase":

        endpoint_consistent = (
            group["delta_A1_A3"] > 0
        )

        # A1 <= A2 <= A3, with an actual endpoint increase.
        monotonic_consistent = (
            (group["A2"] >= group["A1"])
            & (group["A3"] >= group["A2"])
            & (group["A3"] > group["A1"])
        )

    elif dominant_direction == "decrease":

        endpoint_consistent = (
            group["delta_A1_A3"] < 0
        )

        # A1 >= A2 >= A3, with an actual endpoint decrease.
        monotonic_consistent = (
            (group["A2"] <= group["A1"])
            & (group["A3"] <= group["A2"])
            & (group["A3"] < group["A1"])
        )

    else:

        endpoint_consistent = (
            np.isclose(
                group["delta_A1_A3"],
                0,
            )
        )

        monotonic_consistent = (
            np.isclose(group["A1"], group["A2"])
            & np.isclose(group["A2"], group["A3"])
        )


    percent_change_values = (
        group["percent_change_A1_A3"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


    summary_rows.append(
        {
            "exercise": exercise,
            "feature": feature,
            "modality": modality,

            "n_trajectories": len(group),
            "n_true_subjects": group["true_subject"].nunique(),

            "A1_median": group["A1"].median(),
            "A2_median": group["A2"].median(),
            "A3_median": group["A3"].median(),

            "A1_Q1": group["A1"].quantile(0.25),
            "A1_Q3": group["A1"].quantile(0.75),

            "A2_Q1": group["A2"].quantile(0.25),
            "A2_Q3": group["A2"].quantile(0.75),

            "A3_Q1": group["A3"].quantile(0.25),
            "A3_Q3": group["A3"].quantile(0.75),

            "median_delta_A1_A3":
                median_delta,

            "mean_delta_A1_A3":
                group["delta_A1_A3"].mean(),

            "median_percent_change_A1_A3":
                percent_change_values.median()
                if len(percent_change_values) > 0
                else np.nan,

            "dominant_direction":
                dominant_direction,

            "endpoint_direction_consistency":
                endpoint_consistent.mean(),

            "endpoint_direction_consistency_percent":
                endpoint_consistent.mean() * 100.0,

            "monotonic_A1_A2_A3_consistency":
                monotonic_consistent.mean(),

            "monotonic_A1_A2_A3_consistency_percent":
                monotonic_consistent.mean() * 100.0,

            "paired_cohens_dz":
                paired_dz(
                    group["delta_A1_A3"]
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


summary = summary.sort_values(
    [
        "exercise",
        "modality",
        "feature",
    ]
).reset_index(drop=True)


# ============================================================
# KEEP POSE/WATCH AND HR RESULTS SEPARATE
# ============================================================

pose_watch_summary = summary.loc[
    summary["modality"].isin(
        [
            "Pose",
            "Wearable",
        ]
    )
].copy()


hr_summary = summary.loc[
    summary["modality"] == "Physiological"
].copy()


pose_watch_summary.to_csv(
    TABLES_OUT / "02_pose_watch_feature_change_summary.csv",
    index=False,
)


hr_summary.to_csv(
    TABLES_OUT / "02_hr_feature_change_summary.csv",
    index=False,
)


# A combined table is saved ONLY as a convenient master record.
# The cohort difference remains explicit in n_trajectories.

summary.to_csv(
    TABLES_OUT / "02_all_feature_change_summary.csv",
    index=False,
)


# ============================================================
# COHORT CHECK TABLE
# ============================================================

cohort_check = (
    summary[
        [
            "exercise",
            "modality",
            "n_trajectories",
            "n_true_subjects",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "exercise",
            "modality",
        ]
    )
)


cohort_check.to_csv(
    TABLES_OUT / "02_analysis_cohort_check.csv",
    index=False,
)


# ============================================================
# SIMPLE TOP-LINE TERMINAL OUTPUT
# ============================================================

print("\n" + "-" * 80)
print("POSE/WATCH COHORT")
print("-" * 80)

pose_watch_cohort = (
    pose_watch_summary[
        [
            "exercise",
            "n_trajectories",
            "n_true_subjects",
        ]
    ]
    .drop_duplicates()
    .sort_values("exercise")
)

print(
    pose_watch_cohort.to_string(
        index=False
    )
)


print("\n" + "-" * 80)
print("HEART-RATE COHORT — KEPT SEPARATE")
print("-" * 80)

hr_cohort = (
    hr_summary[
        [
            "exercise",
            "n_trajectories",
            "n_true_subjects",
        ]
    ]
    .drop_duplicates()
    .sort_values("exercise")
)

print(
    hr_cohort.to_string(
        index=False
    )
)


print("\n" + "-" * 80)
print("HEART-RATE A1 -> A2 -> A3 DESCRIPTIVE SUMMARY")
print("-" * 80)

hr_display_columns = [
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
        hr_display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# METHOD LOG
# ============================================================

method_log = """
SCRIPT 02 — REPEATED-PERFORMANCE CHANGE ANALYSIS

Primary unit:
One workout_session + exercise trajectory containing A1, A2 and A3.

POSE/WATCH COHORT:
All complete A1/A2/A3 trajectories from Script 01.

HEART-RATE COHORT:
Only complete A1/A2/A3 trajectories with hr_mean_combined available
at all three attempts.

Metrics:
1. A1/A2/A3 median and IQR.
2. Paired A3-A1 change.
3. Paired percentage change: ((A3-A1)/A1)*100.
4. Dominant direction determined from sign of median paired A3-A1 change.
5. Endpoint directional consistency:
   proportion of trajectories changing in the dominant endpoint direction.
6. Monotonic consistency:
   for an increase: A1 <= A2 <= A3 and A3 > A1;
   for a decrease: A1 >= A2 >= A3 and A3 < A1.
7. Paired Cohen's dz:
   mean(A3-A1) / SD(A3-A1).

Interpretation:
These metrics describe repeated-performance change.
They do not independently identify fatigue, engagement, frustration,
clinical deterioration, or any other behavioural/clinical state.

No significance tests are performed in Script 02.
"""

(LOGS_OUT / "02_method_record.txt").write_text(
    method_log.strip(),
    encoding="utf-8",
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("SCRIPT 02 COMPLETE")
print("=" * 80)

print("\nSaved trajectory-level data:")
print(
    DATA_OUT
    / "02_trajectory_level_feature_changes.csv"
)

print("\nSaved pose/watch results:")
print(
    TABLES_OUT
    / "02_pose_watch_feature_change_summary.csv"
)

print("\nSaved HR results:")
print(
    TABLES_OUT
    / "02_hr_feature_change_summary.csv"
)

print("\nSaved combined master record:")
print(
    TABLES_OUT
    / "02_all_feature_change_summary.csv"
)

print("\nSaved cohort check:")
print(
    TABLES_OUT
    / "02_analysis_cohort_check.csv"
)

print("\nSaved method record:")
print(
    LOGS_OUT
    / "02_method_record.txt"
)

print(
    "\nHeart rate remains a separate physiological analysis "
    "so its effect can be examined explicitly later."
)
