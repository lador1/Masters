from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


# ============================================================
# PATHS
# ============================================================

MASTER_PATH = Path(
    r"C:\Masters\data\features\mmfit"
    r"\mmfit_master_segment_table.csv"
)

ATTEMPT_PATH = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\squat_longitudinal_change_dataset.csv"
)

OUTPUT_DIR = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\all_feature_screen"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAILED_DATASET_OUTPUT = (
    OUTPUT_DIR / "all_detailed_squat_features.csv"
)

RANKING_OUTPUT = (
    OUTPUT_DIR / "all_features_progression_ranked.csv"
)

MEDIANS_OUTPUT = (
    OUTPUT_DIR / "all_features_attempt_medians.csv"
)

CHANGES_OUTPUT = (
    OUTPUT_DIR / "all_features_participant_changes.csv"
)

TOP_MODALITY_OUTPUT = (
    OUTPUT_DIR / "top_features_by_modality.csv"
)


# ============================================================
# JOINT NAMES
#
# Current interpreted COCO-style 17-joint layout.
# ============================================================

JOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


# ============================================================
# HELPERS
# ============================================================

def feature_modality(feature):
    """
    Assign each signal feature to a modality.
    """

    f = feature.lower()

    if f.startswith("joint_") or f.startswith("pose_"):
        return "pose"

    if (
        f.startswith("sw_l_acc_")
        or f.startswith("sw_r_acc_")
    ):
        return "watch_accelerometer"

    if (
        f.startswith("sw_l_gyr_")
        or f.startswith("sw_r_gyr_")
    ):
        return "watch_gyroscope"

    if (
        f.startswith("sw_l_hr_")
        or f.startswith("sw_r_hr_")
        or f.startswith("hr_")
    ):
        return "heart_rate"

    return "other"


def readable_feature_name(feature):
    """
    Replace joint_XX with the interpreted body-joint name.
    """

    match = re.match(r"joint_(\d{2})_(.+)", feature)

    if match:
        joint_number = int(match.group(1))
        metric = match.group(2)

        joint_name = JOINT_NAMES.get(
            joint_number,
            f"joint_{joint_number:02d}"
        )

        return f"{joint_name}_{metric}"

    return feature


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg false-discovery-rate correction.
    """

    p = np.asarray(p_values, dtype=float)

    q = np.full(len(p), np.nan)

    valid = np.isfinite(p)

    if valid.sum() == 0:
        return q

    valid_p = p[valid]

    order = np.argsort(valid_p)

    ranked_p = valid_p[order]

    n = len(ranked_p)

    adjusted = ranked_p * n / np.arange(1, n + 1)

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(adjusted, 0, 1)

    valid_indices = np.where(valid)[0]

    q[valid_indices[order]] = adjusted

    return q


def safe_wilcoxon(a, b):
    """
    Paired Wilcoxon test.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    a = a[mask]
    b = b[mask]

    if len(a) < 5:
        return np.nan

    differences = b - a

    if np.allclose(differences, 0):
        return 1.0

    try:
        result = wilcoxon(
            a,
            b,
            alternative="two-sided"
        )
        return float(result.pvalue)

    except ValueError:
        return np.nan


def paired_cohens_dz(a, b):
    """
    Standardised paired change.

    Positive = feature increased from Attempt 1 to Attempt 3.
    Negative = feature decreased.

    Absolute magnitude allows features with different physical
    units to be compared more fairly.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)

    delta = b[mask] - a[mask]

    if len(delta) < 2:
        return np.nan

    sd = np.std(delta, ddof=1)

    if sd == 0 or not np.isfinite(sd):
        return np.nan

    return float(np.mean(delta) / sd)


# ============================================================
# LOAD
# ============================================================

master = pd.read_csv(MASTER_PATH)
attempts = pd.read_csv(ATTEMPT_PATH)

print("Master dataset:")
print(MASTER_PATH)
print("Shape:", master.shape)

print("\nExisting longitudinal cohort:")
print(ATTEMPT_PATH)
print("Shape:", attempts.shape)


# ============================================================
# EXACT ATTEMPT ROWS WE ALREADY VALIDATED
#
# Use the existing 39-row repeated-attempt cohort rather than
# creating a different participant sample.
# ============================================================

key_columns = [
    "participant",
    "start_time",
    "end_time",
]

required_attempt_columns = key_columns + [
    "squat_segment_number"
]

missing_attempt = [
    c for c in required_attempt_columns
    if c not in attempts.columns
]

if missing_attempt:
    raise ValueError(
        f"Longitudinal dataset missing: {missing_attempt}"
    )

missing_master = [
    c for c in key_columns
    if c not in master.columns
]

if missing_master:
    raise ValueError(
        f"Master dataset missing: {missing_master}"
    )


# Keep only our validated attempt identifiers.
attempt_keys = attempts[
    required_attempt_columns
].drop_duplicates()


# ============================================================
# MERGE WITH FULL DETAILED MASTER FEATURES
# ============================================================

detailed = attempt_keys.merge(
    master,
    on=key_columns,
    how="left",
    validate="one_to_one",
    indicator=True,
)

unmatched = detailed["_merge"].ne("both").sum()

print("\nRows not matched to master table:")
print(unmatched)

if unmatched:
    print(
        detailed.loc[
            detailed["_merge"].ne("both"),
            required_attempt_columns
        ].to_string(index=False)
    )

    raise ValueError(
        "Some longitudinal rows could not be matched "
        "to the master feature table."
    )

detailed = detailed.drop(columns="_merge")

detailed = detailed.sort_values(
    ["participant", "squat_segment_number"]
).reset_index(drop=True)


# ============================================================
# IDENTIFY ALL GENUINE SENSOR FEATURES
#
# We deliberately exclude:
# - identifiers
# - duration / pose_frames
# - repetitions
# - validity/count flags
# - HR quality flags
#
# because these should not be allowed to masquerade as
# movement deterioration.
# ============================================================

exclude_exact = {
    "participant",
    "exercise",
    "start_time",
    "end_time",
    "duration",
    "duration_frames",
    "duration_seconds",
    "reps",
    "squat_segment_number",

    "pose_frames",
    "pose_valid",
    "pose_joint_count",

    "hr_left_available",
    "hr_right_available",
    "hr_any_available",
    "hr_both_available",
    "hr_disagreement_flag",
    "hr_left_right_diff",
}

exclude_substrings = [
    "_count",
    "_valid",
    "quality_flag",
]

candidate_features = []

for column in detailed.columns:

    if column in exclude_exact:
        continue

    if any(
        token in column.lower()
        for token in exclude_substrings
    ):
        continue

    modality = feature_modality(column)

    # Only sensor-derived features.
    if modality == "other":
        continue

    numeric = pd.to_numeric(
        detailed[column],
        errors="coerce"
    )

    # Require some usable measurements.
    if numeric.notna().sum() < 5:
        continue

    # Remove constants.
    if numeric.nunique(dropna=True) <= 1:
        continue

    detailed[column] = numeric

    candidate_features.append(column)


print("\n========================================")
print("FEATURE DISCOVERY")
print("========================================")

print("Total detailed sensor features:")
print(len(candidate_features))

for modality in [
    "pose",
    "watch_accelerometer",
    "watch_gyroscope",
    "heart_rate",
]:

    count = sum(
        feature_modality(f) == modality
        for f in candidate_features
    )

    print(f"{modality}: {count}")


# ============================================================
# SAVE FULL DETAILED ATTEMPT TABLE
# ============================================================

save_columns = (
    required_attempt_columns
    + candidate_features
)

detailed[
    save_columns
].to_csv(
    DETAILED_DATASET_OUTPUT,
    index=False
)


# ============================================================
# MEDIANS BY ATTEMPT
# ============================================================

median_rows = []

for feature in candidate_features:

    for attempt_number in [1, 2, 3]:

        values = detailed.loc[
            detailed["squat_segment_number"]
            == attempt_number,
            feature
        ].dropna()

        median_rows.append({
            "feature": feature,
            "feature_label":
                readable_feature_name(feature),
            "modality":
                feature_modality(feature),
            "attempt":
                attempt_number,
            "n":
                len(values),
            "mean":
                values.mean(),
            "median":
                values.median(),
            "std":
                values.std(),
        })

median_table = pd.DataFrame(median_rows)

median_table.to_csv(
    MEDIANS_OUTPUT,
    index=False
)


# ============================================================
# PARTICIPANT-LEVEL ATTEMPT 1 -> 3 CHANGES
# ============================================================

change_rows = []

for participant, group in detailed.groupby(
    "participant"
):

    group = group.set_index(
        "squat_segment_number"
    )

    if 1 not in group.index or 3 not in group.index:
        continue

    for feature in candidate_features:

        value1 = group.loc[1, feature]
        value3 = group.loc[3, feature]

        if (
            pd.isna(value1)
            or pd.isna(value3)
        ):
            continue

        delta = value3 - value1

        if abs(value1) > 1e-8:
            pct_change = (
                delta / abs(value1)
            ) * 100.0
        else:
            pct_change = np.nan

        if delta > 0:
            direction = "increase"

        elif delta < 0:
            direction = "decrease"

        else:
            direction = "no_change"

        change_rows.append({
            "participant":
                participant,
            "feature":
                feature,
            "feature_label":
                readable_feature_name(feature),
            "modality":
                feature_modality(feature),
            "attempt1":
                value1,
            "attempt3":
                value3,
            "delta":
                delta,
            "pct_change":
                pct_change,
            "direction":
                direction,
        })


changes = pd.DataFrame(change_rows)

changes.to_csv(
    CHANGES_OUTPUT,
    index=False
)


# ============================================================
# FEATURE-BY-FEATURE PROGRESSION ANALYSIS
# ============================================================

ranking_rows = []

for feature in candidate_features:

    pivot = detailed.pivot(
        index="participant",
        columns="squat_segment_number",
        values=feature,
    )

    # ----------------------------------------
    # Attempt 1 vs Attempt 3 paired data
    # ----------------------------------------

    pair13 = pivot[
        [1, 3]
    ].dropna() if (
        1 in pivot.columns
        and 3 in pivot.columns
    ) else pd.DataFrame()

    if len(pair13) >= 2:

        a1 = pair13[1].to_numpy(
            dtype=float
        )

        a3 = pair13[3].to_numpy(
            dtype=float
        )

        delta = a3 - a1

        increased = int(
            np.sum(delta > 0)
        )

        decreased = int(
            np.sum(delta < 0)
        )

        unchanged = int(
            np.sum(delta == 0)
        )

        dominant_count = max(
            increased,
            decreased
        )

        direction_consistency = (
            dominant_count
            / len(delta)
        )

        dominant_direction = (
            "increase"
            if increased > decreased
            else "decrease"
            if decreased > increased
            else "mixed"
        )

        median_delta = float(
            np.median(delta)
        )

        mean_delta = float(
            np.mean(delta)
        )

        median_a1 = float(
            np.median(a1)
        )

        median_a3 = float(
            np.median(a3)
        )

        if abs(median_a1) > 1e-8:
            median_change_pct = (
                (
                    median_a3
                    - median_a1
                )
                / abs(median_a1)
            ) * 100
        else:
            median_change_pct = np.nan

        dz = paired_cohens_dz(
            a1,
            a3
        )

        wilcoxon_p = safe_wilcoxon(
            a1,
            a3
        )

    else:

        increased = np.nan
        decreased = np.nan
        unchanged = np.nan

        direction_consistency = np.nan
        dominant_direction = np.nan

        median_delta = np.nan
        mean_delta = np.nan
        median_a1 = np.nan
        median_a3 = np.nan
        median_change_pct = np.nan
        dz = np.nan
        wilcoxon_p = np.nan


    # ----------------------------------------
    # Attempt 1, 2, 3 Friedman test
    # ----------------------------------------

    if all(
        attempt in pivot.columns
        for attempt in [1, 2, 3]
    ):

        complete = pivot[
            [1, 2, 3]
        ].dropna()

        if len(complete) >= 5:

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "ignore"
                    )

                    friedman_result = (
                        friedmanchisquare(
                            complete[1],
                            complete[2],
                            complete[3],
                        )
                    )

                friedman_stat = float(
                    friedman_result.statistic
                )

                friedman_p = float(
                    friedman_result.pvalue
                )

            except ValueError:
                friedman_stat = np.nan
                friedman_p = np.nan

        else:
            friedman_stat = np.nan
            friedman_p = np.nan

        n_complete_123 = len(complete)

    else:
        friedman_stat = np.nan
        friedman_p = np.nan
        n_complete_123 = 0


    # ----------------------------------------
    # Participant-specific slope 1 -> 2 -> 3
    # ----------------------------------------

    slopes = []

    for _, row in pivot.iterrows():

        xs = []
        ys = []

        for attempt_number in [1, 2, 3]:

            if (
                attempt_number
                in row.index
                and pd.notna(
                    row[attempt_number]
                )
            ):
                xs.append(
                    attempt_number
                )
                ys.append(
                    row[attempt_number]
                )

        if len(xs) >= 2:

            slope = np.polyfit(
                xs,
                ys,
                1
            )[0]

            slopes.append(slope)

    slopes = np.asarray(
        slopes,
        dtype=float
    )

    if len(slopes):

        median_slope = float(
            np.median(slopes)
        )

        positive_slopes = int(
            np.sum(slopes > 0)
        )

        negative_slopes = int(
            np.sum(slopes < 0)
        )

        slope_consistency = (
            max(
                positive_slopes,
                negative_slopes
            )
            / len(slopes)
        )

    else:

        median_slope = np.nan
        positive_slopes = np.nan
        negative_slopes = np.nan
        slope_consistency = np.nan


    ranking_rows.append({
        "feature":
            feature,

        "feature_label":
            readable_feature_name(
                feature
            ),

        "modality":
            feature_modality(feature),

        "n_pairs_1_3":
            len(pair13),

        "n_complete_1_2_3":
            n_complete_123,

        "attempt1_median":
            median_a1,

        "attempt3_median":
            median_a3,

        "median_delta_1_to_3":
            median_delta,

        "mean_delta_1_to_3":
            mean_delta,

        "median_pct_change_1_to_3":
            median_change_pct,

        "participants_increased":
            increased,

        "participants_decreased":
            decreased,

        "participants_unchanged":
            unchanged,

        "dominant_direction":
            dominant_direction,

        "direction_consistency":
            direction_consistency,

        "paired_cohens_dz":
            dz,

        "abs_paired_cohens_dz":
            abs(dz)
            if pd.notna(dz)
            else np.nan,

        "wilcoxon_p":
            wilcoxon_p,

        "friedman_stat":
            friedman_stat,

        "friedman_p":
            friedman_p,

        "median_attempt_slope":
            median_slope,

        "positive_slopes":
            positive_slopes,

        "negative_slopes":
            negative_slopes,

        "slope_consistency":
            slope_consistency,
    })


ranking = pd.DataFrame(
    ranking_rows
)


# ============================================================
# MULTIPLE-COMPARISON CORRECTION
#
# We are testing many features, so raw p-values alone would
# generate false positives.
# ============================================================

ranking["wilcoxon_fdr_q"] = (
    benjamini_hochberg(
        ranking["wilcoxon_p"]
    )
)

ranking["friedman_fdr_q"] = (
    benjamini_hochberg(
        ranking["friedman_p"]
    )
)


# ============================================================
# PRIMARY RANKING
#
# Main sort:
# 1. magnitude of standardised Attempt 1 -> 3 change
# 2. consistency across participants
#
# Statistical significance remains visible separately.
# ============================================================

ranking = ranking.sort_values(
    by=[
        "abs_paired_cohens_dz",
        "direction_consistency",
    ],
    ascending=[
        False,
        False,
    ],
    na_position="last",
).reset_index(drop=True)


ranking["overall_rank"] = (
    np.arange(
        1,
        len(ranking) + 1
    )
)


ranking.to_csv(
    RANKING_OUTPUT,
    index=False
)


# ============================================================
# TOP FEATURES WITHIN EACH MODALITY
# ============================================================

top_modality_tables = []

for modality in [
    "pose",
    "watch_accelerometer",
    "watch_gyroscope",
    "heart_rate",
]:

    subset = (
        ranking[
            ranking["modality"]
            == modality
        ]
        .head(20)
        .copy()
    )

    subset["modality_rank"] = (
        np.arange(
            1,
            len(subset) + 1
        )
    )

    top_modality_tables.append(
        subset
    )

top_by_modality = pd.concat(
    top_modality_tables,
    ignore_index=True,
)

top_by_modality.to_csv(
    TOP_MODALITY_OUTPUT,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("TOP 30 FEATURES OVERALL")
print("========================================")

display_columns = [
    "overall_rank",
    "feature_label",
    "modality",
    "attempt1_median",
    "attempt3_median",
    "dominant_direction",
    "direction_consistency",
    "paired_cohens_dz",
    "wilcoxon_p",
    "wilcoxon_fdr_q",
    "friedman_p",
    "friedman_fdr_q",
]

print(
    ranking[
        display_columns
    ]
    .head(30)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# PRINT EACH MODALITY
# ============================================================

for modality in [
    "pose",
    "watch_accelerometer",
    "watch_gyroscope",
    "heart_rate",
]:

    print(
        "\n========================================"
    )

    print(
        f"TOP {modality.upper()} FEATURES"
    )

    print(
        "========================================"
    )

    subset = ranking[
        ranking["modality"]
        == modality
    ].head(15)

    print(
        subset[
            [
                "feature_label",
                "dominant_direction",
                "direction_consistency",
                "paired_cohens_dz",
                "wilcoxon_fdr_q",
                "friedman_fdr_q",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


# ============================================================
# FEATURES SURVIVING FDR
# ============================================================

print("\n========================================")
print("FEATURES WITH FDR q < 0.05")
print("========================================")

significant = ranking[
    (
        ranking["wilcoxon_fdr_q"]
        < 0.05
    )
    |
    (
        ranking["friedman_fdr_q"]
        < 0.05
    )
]

if significant.empty:

    print(
        "None survived FDR correction."
    )

else:

    print(
        significant[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )


# ============================================================
# OUTPUTS
# ============================================================

print("\n========================================")
print("SAVED OUTPUTS")
print("========================================")

print(DETAILED_DATASET_OUTPUT)
print(MEDIANS_OUTPUT)
print(CHANGES_OUTPUT)
print(RANKING_OUTPUT)
print(TOP_MODALITY_OUTPUT)

print("\nDone.")