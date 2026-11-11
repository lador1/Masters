from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\full_feature_screen\squat_full_feature_dataset.csv"
)

OUTPUT_DIR = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\candidate_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "squat_candidate_feature_summary.csv"
)

PARTICIPANT_OUTPUT = (
    OUTPUT_DIR
    / "squat_candidate_feature_participant_values.csv"
)


CANDIDATE_FEATURES = {
    "hr_mean_combined":
        "Heart rate mean",

    "knee_to_hip_ratio_median":
        "Knee-to-hip separation ratio",

    "knee_flexion_asymmetry_pct_mean":
        "Knee flexion asymmetry",

    "rep_stance_width_first_to_last_change":
        "Within-set stance-width change",

    "global_hip_root_acceleration_mean":
        "Hip-root acceleration",

    "global_hip_root_jerk_median":
        "Hip-root jerk",

    "sw_l_gyr_mag_min":
        "Left-watch gyroscope minimum",

    "rep_duration_mean":
        "Mean repetition duration",

    "rep_duration_cv_pct":
        "Repetition timing variability",

    "thorax_ldlj":
        "Thorax movement smoothness",
}


data = pd.read_csv(
    INPUT_PATH
)


required = [
    "participant",
    "squat_segment_number",
]


for column in required:

    if column not in data.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


available_features = {}


for feature, label in CANDIDATE_FEATURES.items():

    if feature in data.columns:

        available_features[
            feature
        ] = label

    else:

        print(
            f"Not found, skipped: {feature}"
        )


print("\n========================================")
print("CANDIDATE FEATURES")
print("========================================")

for feature, label in available_features.items():

    print(
        f"{label}: {feature}"
    )


summary_rows = []


for feature, label in available_features.items():

    for attempt in [
        1,
        2,
        3,
    ]:

        values = pd.to_numeric(
            data.loc[
                data[
                    "squat_segment_number"
                ]
                ==
                attempt,
                feature
            ],
            errors="coerce"
        ).dropna()


        if len(values) == 0:
            continue


        summary_rows.append({
            "feature":
                feature,

            "label":
                label,

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

            "q1":
                values.quantile(
                    0.25
                ),

            "q3":
                values.quantile(
                    0.75
                ),

            "iqr":
                (
                    values.quantile(
                        0.75
                    )
                    -
                    values.quantile(
                        0.25
                    )
                ),

            "min":
                values.min(),

            "max":
                values.max(),
        })


summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)


participant_columns = [
    "participant",
    "squat_segment_number",
]


participant_columns += list(
    available_features.keys()
)


participant_values = data[
    participant_columns
].copy()


participant_values.to_csv(
    PARTICIPANT_OUTPUT,
    index=False
)


print("\n========================================")
print("ATTEMPT MEDIANS")
print("========================================")


pivot = summary.pivot(
    index=[
        "feature",
        "label",
    ],
    columns="attempt",
    values="median"
).reset_index()


pivot = pivot.rename(
    columns={
        1: "attempt_1_median",
        2: "attempt_2_median",
        3: "attempt_3_median",
    }
)


if (
    "attempt_1_median"
    in pivot.columns
    and
    "attempt_3_median"
    in pivot.columns
):

    pivot[
        "change_1_to_3"
    ] = (
        pivot[
            "attempt_3_median"
        ]
        -
        pivot[
            "attempt_1_median"
        ]
    )


    pivot[
        "pct_change_1_to_3"
    ] = np.where(
        np.abs(
            pivot[
                "attempt_1_median"
            ]
        )
        >
        1e-12,

        (
            pivot[
                "attempt_3_median"
            ]
            -
            pivot[
                "attempt_1_median"
            ]
        )
        /
        np.abs(
            pivot[
                "attempt_1_median"
            ]
        )
        *
        100.0,

        np.nan
    )


print(
    pivot.round(
        4
    ).to_string(
        index=False
    )
)


pivot.to_csv(
    OUTPUT_DIR
    / "squat_candidate_feature_attempt_comparison.csv",
    index=False
)


# Participant-level line plots

for feature, label in available_features.items():

    plot_data = data[
        [
            "participant",
            "squat_segment_number",
            feature,
        ]
    ].copy()


    plot_data[
        feature
    ] = pd.to_numeric(
        plot_data[
            feature
        ],
        errors="coerce"
    )


    plot_data = plot_data.dropna()


    plt.figure(
        figsize=(9, 6)
    )


    for participant, group in plot_data.groupby(
        "participant"
    ):

        group = group.sort_values(
            "squat_segment_number"
        )

        plt.plot(
            group[
                "squat_segment_number"
            ],
            group[
                feature
            ],
            marker="o",
            alpha=0.45
        )


    medians = (
        plot_data
        .groupby(
            "squat_segment_number"
        )[
            feature
        ]
        .median()
    )


    plt.plot(
        medians.index,
        medians.values,
        marker="o",
        linewidth=4,
        label="Group median"
    )


    plt.xticks(
        [
            1,
            2,
            3,
        ],
        [
            "Attempt 1",
            "Attempt 2",
            "Attempt 3",
        ]
    )


    plt.xlabel(
        "Squat attempt"
    )

    plt.ylabel(
        label
    )

    plt.title(
        f"{label} across repeated squat attempts"
    )

    plt.legend()

    plt.tight_layout()


    safe_name = (
        feature
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
    )


    output_path = (
        OUTPUT_DIR
        / f"{safe_name}_attempt_progression.png"
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print("\n========================================")
print("SAVED")
print("========================================")

print(
    SUMMARY_OUTPUT
)

print(
    PARTICIPANT_OUTPUT
)

print(
    OUTPUT_DIR
    / "squat_candidate_feature_attempt_comparison.csv"
)

print(
    "\nProgression plots saved in:"
)

print(
    OUTPUT_DIR
)

print("\nDone.")