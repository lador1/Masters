from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    friedmanchisquare,
    wilcoxon,
)


# ============================================================
# PATHS
# ============================================================

ADVANCED_PATH = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\advanced_features\squat_advanced_movement_features.csv"
)

MASTER_PATH = Path(
    r"C:\Masters\data\features\mmfit"
    r"\mmfit_master_segment_table.csv"
)

OUTPUT_DIR = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\full_feature_screen"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MERGED_OUTPUT = (
    OUTPUT_DIR
    / "squat_full_feature_dataset.csv"
)

RANKING_OUTPUT = (
    OUTPUT_DIR
    / "squat_full_feature_progression_ranked.csv"
)

CHANGE_OUTPUT = (
    OUTPUT_DIR
    / "squat_full_feature_participant_changes.csv"
)

MEDIAN_OUTPUT = (
    OUTPUT_DIR
    / "squat_full_feature_attempt_medians.csv"
)

MODALITY_OUTPUT = (
    OUTPUT_DIR
    / "squat_top_features_by_modality.csv"
)


# ============================================================
# CORRECT MM-FIT 3D JOINT MAPPING
# ============================================================

JOINT_NAMES = {
    0: "hip_root",

    1: "right_hip",
    2: "right_knee",
    3: "right_foot",

    4: "left_hip",
    5: "left_knee",
    6: "left_foot",

    7: "spine",
    8: "thorax",
    9: "neck_nose",
    10: "head",

    11: "left_shoulder",
    12: "left_elbow",
    13: "left_wrist",

    14: "right_shoulder",
    15: "right_elbow",
    16: "right_wrist",
}


# ============================================================
# HELPERS
# ============================================================

def readable_feature_name(feature):

    match = re.match(
        r"joint_(\d{2})_(.+)",
        feature
    )

    if match:

        joint_number = int(
            match.group(1)
        )

        metric = match.group(2)

        joint_name = JOINT_NAMES.get(
            joint_number,
            f"joint_{joint_number:02d}"
        )

        return (
            f"{joint_name}_{metric}"
        )

    return feature


def feature_modality(feature):

    f = feature.lower()

    if (
        f.startswith("sw_l_acc_")
        or
        f.startswith("sw_r_acc_")
    ):
        return "watch_accelerometer"

    if (
        f.startswith("sw_l_gyr_")
        or
        f.startswith("sw_r_gyr_")
    ):
        return "watch_gyroscope"

    if (
        f.startswith("sw_l_hr_")
        or
        f.startswith("sw_r_hr_")
        or
        f.startswith("hr_")
    ):
        return "heart_rate"

    if (
        f.startswith("joint_")
        or
        f.startswith("pose_")
    ):
        return "pose_original"

    advanced_terms = [
        "knee",
        "hip",
        "thorax",
        "trunk",
        "stance",
        "foot",
        "spine",
        "shoulder",
        "elbow",
        "wrist",
        "head",
        "neck",
        "whole_body",
        "relative_squat",
        "rep_",
        "global_hip_root",
    ]

    if any(
        term in f
        for term in advanced_terms
    ):
        return "pose_advanced"

    return "other"


def feature_family(feature):

    f = feature.lower()

    if "jerk" in f:
        return "jerk"

    if "acceleration" in f:
        return "acceleration"

    if "speed" in f:
        return "speed"

    if "ldlj" in f:
        return "smoothness"

    if "path_efficiency" in f:
        return "path_efficiency"

    if "path_length" in f:
        return "path_length"

    if "asymmetry" in f:
        return "asymmetry"

    if "knee" in f and "angle" in f:
        return "knee_angle"

    if "knee_flexion" in f:
        return "knee_flexion"

    if "hip_angle" in f:
        return "hip_angle"

    if "squat_depth" in f or "rep_depth" in f:
        return "squat_depth"

    if "stance" in f:
        return "stance"

    if "separation" in f:
        return "joint_separation"

    if "trunk" in f:
        return "trunk"

    if "rep_duration" in f:
        return "rep_timing"

    if f.startswith("sw_"):
        return "wearable"

    if f.startswith("hr_"):
        return "heart_rate"

    if f.startswith("joint_"):
        return "original_joint_motion"

    if f.startswith("pose_"):
        return "original_pose_summary"

    return "other"


def benjamini_hochberg(p_values):

    p = np.asarray(
        p_values,
        dtype=float
    )

    result = np.full(
        len(p),
        np.nan
    )

    valid = np.isfinite(p)

    if valid.sum() == 0:
        return result

    valid_p = p[valid]

    order = np.argsort(
        valid_p
    )

    ranked = valid_p[
        order
    ]

    n = len(
        ranked
    )

    adjusted = (
        ranked
        *
        n
        /
        np.arange(
            1,
            n + 1
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1
    )

    valid_indices = np.where(
        valid
    )[0]

    result[
        valid_indices[
            order
        ]
    ] = adjusted

    return result


def paired_cohens_dz(a, b):

    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    mask = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    difference = (
        b[mask]
        -
        a[mask]
    )

    if len(difference) < 2:
        return np.nan

    sd = np.std(
        difference,
        ddof=1
    )

    if (
        not np.isfinite(sd)
        or
        sd <= 1e-12
    ):
        return np.nan

    return float(
        np.mean(difference)
        /
        sd
    )


def safe_wilcoxon(a, b):

    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    mask = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    a = a[mask]
    b = b[mask]

    if len(a) < 5:
        return np.nan

    difference = (
        b - a
    )

    if np.allclose(
        difference,
        0
    ):
        return 1.0

    try:

        result = wilcoxon(
            a,
            b,
            alternative="two-sided"
        )

        return float(
            result.pvalue
        )

    except ValueError:

        return np.nan


# ============================================================
# LOAD
# ============================================================

advanced = pd.read_csv(
    ADVANCED_PATH
)

master = pd.read_csv(
    MASTER_PATH
)


print("\n========================================")
print("INPUT DATA")
print("========================================")

print(
    "Advanced feature table:",
    advanced.shape
)

print(
    "Master feature table:",
    master.shape
)


# ============================================================
# MATCH ADVANCED ATTEMPTS TO MASTER TABLE
# ============================================================

if "start_time" in master.columns:

    master = master.rename(
        columns={
            "start_time":
                "start_frame",

            "end_time":
                "end_frame",
        }
    )


keys = [
    "participant",
    "start_frame",
    "end_frame",
]


for key in keys:

    if key not in advanced.columns:

        raise ValueError(
            f"Advanced table missing key: {key}"
        )

    if key not in master.columns:

        raise ValueError(
            f"Master table missing key: {key}"
        )


advanced[
    "start_frame"
] = pd.to_numeric(
    advanced["start_frame"],
    errors="coerce"
).round().astype("Int64")


advanced[
    "end_frame"
] = pd.to_numeric(
    advanced["end_frame"],
    errors="coerce"
).round().astype("Int64")


master[
    "start_frame"
] = pd.to_numeric(
    master["start_frame"],
    errors="coerce"
).round().astype("Int64")


master[
    "end_frame"
] = pd.to_numeric(
    master["end_frame"],
    errors="coerce"
).round().astype("Int64")


# ============================================================
# AVOID DUPLICATING METADATA
# ============================================================

master_metadata_to_remove = [
    "exercise",
    "reps",
    "duration",
    "duration_frames",
    "duration_seconds",
]


master_for_merge = master.copy()


for column in master_metadata_to_remove:

    if column in master_for_merge.columns:

        master_for_merge = (
            master_for_merge.drop(
                columns=column
            )
        )


duplicate_columns = [
    column
    for column
    in master_for_merge.columns
    if (
        column in advanced.columns
        and
        column not in keys
    )
]


if duplicate_columns:

    master_for_merge = (
        master_for_merge.drop(
            columns=duplicate_columns
        )
    )


merged = advanced.merge(
    master_for_merge,
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True
)


unmatched = (
    merged["_merge"]
    !=
    "both"
).sum()


print(
    "\nUnmatched attempts:",
    unmatched
)


if unmatched > 0:

    print(
        merged.loc[
            merged["_merge"]
            !=
            "both",
            [
                "participant",
                "squat_segment_number",
                "start_frame",
                "end_frame",
            ]
        ].to_string(
            index=False
        )
    )

    raise ValueError(
        "Some advanced attempts could not be matched "
        "to the master segment table."
    )


merged = merged.drop(
    columns="_merge"
)


merged = merged.sort_values(
    [
        "participant",
        "squat_segment_number",
    ]
).reset_index(
    drop=True
)


merged.to_csv(
    MERGED_OUTPUT,
    index=False
)


print(
    "Merged table:",
    merged.shape
)


# ============================================================
# IDENTIFY FEATURES
# ============================================================

exclude_exact = {
    "participant",
    "participant_segment_number",
    "squat_segment_number",

    "start_frame",
    "end_frame",

    "exercise",

    "reps",
    "labelled_reps",

    "duration",
    "duration_frames",
    "duration_seconds",

    "pose_frames",

    "detected_reps",
    "rep_count_difference",

    "rep_period_estimate_frames",
    "rep_period_estimate_seconds",

    "pose_finite_pct",
    "pose_frame_jump_mean",
    "pose_frame_jump_p95",
    "pose_frame_jump_max",

    "hr_left_available",
    "hr_right_available",
    "hr_any_available",
    "hr_both_available",

    "hr_disagreement_flag",
    "hr_left_right_diff",
}


exclude_substrings = [
    "quality_flag",
    "_available",
    "_valid",
    "_count",
]


candidate_features = []


for column in merged.columns:

    if column in exclude_exact:
        continue

    if any(
        token in column.lower()
        for token in exclude_substrings
    ):
        continue

    # Exclude raw sample-count columns such as sw_l_hr_n.
    if re.search(
        r"_n$",
        column.lower()
    ):
        continue

    numeric = pd.to_numeric(
        merged[column],
        errors="coerce"
    )

    if numeric.notna().sum() < 5:
        continue

    if numeric.nunique(
        dropna=True
    ) <= 1:
        continue

    merged[
        column
    ] = numeric

    candidate_features.append(
        column
    )


print("\n========================================")
print("FEATURE BANK")
print("========================================")

print(
    "Features to screen:",
    len(
        candidate_features
    )
)


modality_counts = (
    pd.Series(
        [
            feature_modality(
                feature
            )
            for feature
            in candidate_features
        ]
    )
    .value_counts()
)


print(
    modality_counts.to_string()
)


# ============================================================
# ATTEMPT MEDIANS
# ============================================================

median_rows = []


for feature in candidate_features:

    for attempt in [
        1,
        2,
        3,
    ]:

        values = merged.loc[
            merged[
                "squat_segment_number"
            ]
            ==
            attempt,
            feature
        ].dropna()


        median_rows.append({
            "feature":
                feature,

            "feature_label":
                readable_feature_name(
                    feature
                ),

            "modality":
                feature_modality(
                    feature
                ),

            "feature_family":
                feature_family(
                    feature
                ),

            "attempt":
                attempt,

            "n":
                len(values),

            "mean":
                values.mean(),

            "median":
                values.median(),

            "std":
                values.std(),

            "min":
                values.min(),

            "max":
                values.max(),
        })


median_df = pd.DataFrame(
    median_rows
)


median_df.to_csv(
    MEDIAN_OUTPUT,
    index=False
)


# ============================================================
# PARTICIPANT-LEVEL CHANGES
# ============================================================

change_rows = []


for participant, group in merged.groupby(
    "participant"
):

    group = group.set_index(
        "squat_segment_number"
    )


    if (
        1 not in group.index
        or
        3 not in group.index
    ):
        continue


    for feature in candidate_features:

        value1 = group.loc[
            1,
            feature
        ]

        value3 = group.loc[
            3,
            feature
        ]


        if (
            pd.isna(value1)
            or
            pd.isna(value3)
        ):
            continue


        delta = (
            value3
            -
            value1
        )


        if abs(
            value1
        ) > 1e-12:

            pct_change = (
                delta
                /
                abs(value1)
                *
                100.0
            )

        else:

            pct_change = np.nan


        if delta > 0:

            direction = (
                "increase"
            )

        elif delta < 0:

            direction = (
                "decrease"
            )

        else:

            direction = (
                "no_change"
            )


        change_rows.append({
            "participant":
                participant,

            "feature":
                feature,

            "feature_label":
                readable_feature_name(
                    feature
                ),

            "modality":
                feature_modality(
                    feature
                ),

            "feature_family":
                feature_family(
                    feature
                ),

            "attempt1_value":
                value1,

            "attempt3_value":
                value3,

            "delta":
                delta,

            "pct_change":
                pct_change,

            "direction":
                direction,
        })


changes = pd.DataFrame(
    change_rows
)


changes.to_csv(
    CHANGE_OUTPUT,
    index=False
)


# ============================================================
# FEATURE PROGRESSION ANALYSIS
# ============================================================

ranking_rows = []


for feature in candidate_features:

    pivot = merged.pivot(
        index="participant",
        columns="squat_segment_number",
        values=feature
    )


    # ----------------------------------------
    # ATTEMPT 1 VS 3
    # ----------------------------------------

    if (
        1 in pivot.columns
        and
        3 in pivot.columns
    ):

        pair13 = pivot[
            [
                1,
                3,
            ]
        ].dropna()

    else:

        pair13 = pd.DataFrame()


    if len(
        pair13
    ) >= 2:

        a1 = pair13[
            1
        ].to_numpy(
            dtype=float
        )

        a3 = pair13[
            3
        ].to_numpy(
            dtype=float
        )


        delta = (
            a3 - a1
        )


        increased = int(
            np.sum(
                delta > 0
            )
        )

        decreased = int(
            np.sum(
                delta < 0
            )
        )

        unchanged = int(
            np.sum(
                delta == 0
            )
        )


        dominant_count = max(
            increased,
            decreased
        )


        direction_consistency = (
            dominant_count
            /
            len(delta)
        )


        if increased > decreased:

            dominant_direction = (
                "increase"
            )

        elif decreased > increased:

            dominant_direction = (
                "decrease"
            )

        else:

            dominant_direction = (
                "mixed"
            )


        median_a1 = float(
            np.median(
                a1
            )
        )

        median_a3 = float(
            np.median(
                a3
            )
        )


        median_delta = float(
            np.median(
                delta
            )
        )

        mean_delta = float(
            np.mean(
                delta
            )
        )


        if abs(
            median_a1
        ) > 1e-12:

            median_pct_change = (
                (
                    median_a3
                    -
                    median_a1
                )
                /
                abs(
                    median_a1
                )
                *
                100.0
            )

        else:

            median_pct_change = (
                np.nan
            )


        dz = paired_cohens_dz(
            a1,
            a3
        )


        wilcoxon_p = safe_wilcoxon(
            a1,
            a3
        )


        participant_pct_changes = []

        for first, third in zip(
            a1,
            a3
        ):

            if abs(
                first
            ) > 1e-12:

                participant_pct_changes.append(
                    (
                        third
                        -
                        first
                    )
                    /
                    abs(first)
                    *
                    100.0
                )


        if participant_pct_changes:

            median_participant_pct_change = float(
                np.median(
                    participant_pct_changes
                )
            )

            max_abs_participant_pct_change = float(
                np.max(
                    np.abs(
                        participant_pct_changes
                    )
                )
            )

        else:

            median_participant_pct_change = (
                np.nan
            )

            max_abs_participant_pct_change = (
                np.nan
            )


    else:

        increased = np.nan
        decreased = np.nan
        unchanged = np.nan

        direction_consistency = (
            np.nan
        )

        dominant_direction = (
            np.nan
        )

        median_a1 = np.nan
        median_a3 = np.nan

        median_delta = np.nan
        mean_delta = np.nan

        median_pct_change = (
            np.nan
        )

        median_participant_pct_change = (
            np.nan
        )

        max_abs_participant_pct_change = (
            np.nan
        )

        dz = np.nan
        wilcoxon_p = np.nan


    # ----------------------------------------
    # ATTEMPTS 1, 2, 3 FRIEDMAN
    # ----------------------------------------

    if all(
        attempt in pivot.columns
        for attempt in [
            1,
            2,
            3,
        ]
    ):

        complete = pivot[
            [
                1,
                2,
                3,
            ]
        ].dropna()


        n_complete = len(
            complete
        )


        if n_complete >= 5:

            try:

                with warnings.catch_warnings():

                    warnings.simplefilter(
                        "ignore"
                    )

                    result = friedmanchisquare(
                        complete[1],
                        complete[2],
                        complete[3],
                    )


                friedman_stat = float(
                    result.statistic
                )

                friedman_p = float(
                    result.pvalue
                )


            except ValueError:

                friedman_stat = np.nan
                friedman_p = np.nan


        else:

            friedman_stat = np.nan
            friedman_p = np.nan


    else:

        n_complete = 0

        friedman_stat = np.nan
        friedman_p = np.nan


    # ----------------------------------------
    # PARTICIPANT-SPECIFIC SLOPE
    # ----------------------------------------

    slopes = []


    for _, participant_row in pivot.iterrows():

        xs = []
        ys = []


        for attempt in [
            1,
            2,
            3,
        ]:

            if (
                attempt
                in participant_row.index
                and
                pd.notna(
                    participant_row[
                        attempt
                    ]
                )
            ):

                xs.append(
                    attempt
                )

                ys.append(
                    participant_row[
                        attempt
                    ]
                )


        if len(xs) >= 2:

            slope = np.polyfit(
                xs,
                ys,
                1
            )[0]

            slopes.append(
                slope
            )


    slopes = np.asarray(
        slopes,
        dtype=float
    )


    if len(slopes) > 0:

        median_slope = float(
            np.median(
                slopes
            )
        )

        positive_slopes = int(
            np.sum(
                slopes > 0
            )
        )

        negative_slopes = int(
            np.sum(
                slopes < 0
            )
        )


        slope_consistency = (
            max(
                positive_slopes,
                negative_slopes
            )
            /
            len(slopes)
        )


    else:

        median_slope = np.nan

        positive_slopes = np.nan
        negative_slopes = np.nan

        slope_consistency = (
            np.nan
        )


    ranking_rows.append({
        "feature":
            feature,

        "feature_label":
            readable_feature_name(
                feature
            ),

        "modality":
            feature_modality(
                feature
            ),

        "feature_family":
            feature_family(
                feature
            ),

        "n_pairs_1_3":
            len(
                pair13
            ),

        "n_complete_1_2_3":
            n_complete,

        "attempt1_median":
            median_a1,

        "attempt3_median":
            median_a3,

        "median_delta_1_to_3":
            median_delta,

        "mean_delta_1_to_3":
            mean_delta,

        "median_pct_change_from_group_medians":
            median_pct_change,

        "median_participant_pct_change":
            median_participant_pct_change,

        "max_abs_participant_pct_change":
            max_abs_participant_pct_change,

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
# FDR CORRECTION
# ============================================================

ranking[
    "wilcoxon_fdr_q"
] = benjamini_hochberg(
    ranking[
        "wilcoxon_p"
    ]
)


ranking[
    "friedman_fdr_q"
] = benjamini_hochberg(
    ranking[
        "friedman_p"
    ]
)


# ============================================================
# RANK FEATURES
# ============================================================

ranking = ranking.sort_values(
    by=[
        "abs_paired_cohens_dz",
        "direction_consistency",
        "slope_consistency",
    ],
    ascending=[
        False,
        False,
        False,
    ],
    na_position="last"
).reset_index(
    drop=True
)


ranking[
    "overall_rank"
] = np.arange(
    1,
    len(ranking) + 1
)


ranking.to_csv(
    RANKING_OUTPUT,
    index=False
)


# ============================================================
# TOP FEATURES BY MODALITY
# ============================================================

modality_tables = []


for modality in sorted(
    ranking[
        "modality"
    ].dropna().unique()
):

    subset = (
        ranking[
            ranking[
                "modality"
            ]
            ==
            modality
        ]
        .head(25)
        .copy()
    )


    subset[
        "modality_rank"
    ] = np.arange(
        1,
        len(subset) + 1
    )


    modality_tables.append(
        subset
    )


if modality_tables:

    modality_output = pd.concat(
        modality_tables,
        ignore_index=True
    )

else:

    modality_output = (
        pd.DataFrame()
    )


modality_output.to_csv(
    MODALITY_OUTPUT,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

display_columns = [
    "overall_rank",

    "feature_label",

    "modality",

    "feature_family",

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


print("\n========================================")
print("TOP 40 FEATURES OVERALL")
print("========================================")

print(
    ranking[
        display_columns
    ]
    .head(40)
    .round(5)
    .to_string(
        index=False
    )
)


# ============================================================
# FDR RESULTS
# ============================================================

print("\n========================================")
print("FEATURES SURVIVING FDR q < 0.05")
print("========================================")


significant = ranking[
    (
        ranking[
            "wilcoxon_fdr_q"
        ]
        <
        0.05
    )
    |
    (
        ranking[
            "friedman_fdr_q"
        ]
        <
        0.05
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
        .round(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# TOP BY MODALITY
# ============================================================

for modality in [
    "pose_advanced",
    "pose_original",
    "watch_accelerometer",
    "watch_gyroscope",
    "heart_rate",
]:

    subset = ranking[
        ranking[
            "modality"
        ]
        ==
        modality
    ].head(
        15
    )


    if subset.empty:
        continue


    print(
        "\n========================================"
    )

    print(
        f"TOP {modality.upper()}"
    )

    print(
        "========================================"
    )


    print(
        subset[
            [
                "feature_label",
                "feature_family",

                "dominant_direction",

                "direction_consistency",

                "paired_cohens_dz",

                "wilcoxon_fdr_q",

                "friedman_fdr_q",
            ]
        ]
        .round(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

print("\n========================================")
print("COMPLETE")
print("========================================")

print(
    "Participants:",
    merged[
        "participant"
    ].nunique()
)

print(
    "Attempts:",
    len(
        merged
    )
)

print(
    "Features screened:",
    len(
        ranking
    )
)


print("\nSaved:")

print(
    MERGED_OUTPUT
)

print(
    RANKING_OUTPUT
)

print(
    CHANGE_OUTPUT
)

print(
    MEDIAN_OUTPUT
)

print(
    MODALITY_OUTPUT
)

print("\nDone.")