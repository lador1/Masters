"""
07_test_multimodal_fusion.py

PRIMARY THESIS ANALYSIS — MULTIMODAL FUSION

Purpose
-------
Test whether combining Pose, Wearable and Heart-Rate features improves
discrimination between the first (A1) and third (A3) repeated attempts.

This is the first script that actually tests multimodal fusion.

IMPORTANT
---------
Previous scripts measured feature responsiveness.

This script asks a different question:

    Does combining modalities provide more predictive information
    about repeated-performance stage than individual modalities?

Target
------
A1 = early repeated attempt
A3 = late repeated attempt

A2 is excluded from this binary classification experiment.

The target is NOT fatigue.

Validation
----------
Leave-One-True-Subject-Out (LOSO).

All workout sessions belonging to the held-out subject remain entirely
outside training.

This evaluates generalisation to an unseen person.

Model
-----
Random Forest classifier.

The Random Forest is used as a fixed baseline model consistent with
the earlier MM-Fit classification experiments.

No hyperparameter optimisation is performed in this script.

Subjects with multiple training workout sessions are sample-weighted
so that each true subject contributes approximately equal total weight.

Primary metrics
---------------
Balanced accuracy
Macro F1
ROC-AUC

Fusion gain
-----------
Fusion gain is calculated as a simple difference in predictive
performance.

Example:

Pose+Watch gain =
    Pose+Watch performance - max(Pose, Watch)

This is NOT a new sensitivity score.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut


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
LOGS_OUT = RESULTS_ROOT / "logs"


for folder in [
    DATA_OUT,
    TABLES_OUT,
    LOGS_OUT,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# FEATURE SETS
# ============================================================

POSE = [
    "pose_movement_mean",
    "pose_movement_std",
    "pose_movement_max",
    "pose_range_mean",
]


WEARABLE = [
    "watch_acc_mag_mean",
    "watch_acc_mag_std",
    "watch_gyr_mag_mean",
    "watch_gyr_mag_std",
]


HR = [
    "hr_mean_combined",
]


FEATURE_SETS = {

    "Pose":
        POSE,

    "Wearable":
        WEARABLE,

    "HR":
        HR,

    "Pose+Wearable":
        POSE
        + WEARABLE,

    "Pose+HR":
        POSE
        + HR,

    "Wearable+HR":
        WEARABLE
        + HR,

    "Pose+Wearable+HR":
        POSE
        + WEARABLE
        + HR,
}


# ============================================================
# FIXED RANDOM FOREST SETTINGS
# ============================================================

RF_PARAMETERS = {

    "n_estimators":
        500,

    "random_state":
        42,

    "n_jobs":
        -1,

    "class_weight":
        "balanced_subsample",

    "max_features":
        "sqrt",
}


# ============================================================
# LOAD
# ============================================================

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        "\nMatched multimodal cohort not found:\n"
        f"{INPUT_PATH}"
    )


df = pd.read_csv(
    INPUT_PATH
)


print("\n" + "=" * 88)
print("MM-FIT REPEATED PERFORMANCE ANALYSIS")
print("SCRIPT 07 — MULTIMODAL FUSION: A1 VS A3")
print("=" * 88)


print(
    f"\nInput:\n{INPUT_PATH}"
)


print(
    f"\nRows loaded: {len(df)}"
)


print(
    f"True subjects: {df['true_subject'].nunique()}"
)


print(
    f"Exercises: {df['exercise'].nunique()}"
)


# ============================================================
# KEEP A1 AND A3 ONLY
# ============================================================

analysis = (
    df.loc[
        df[
            "attempt"
        ].isin(
            [
                "A1",
                "A3",
            ]
        )
    ]
    .copy()
)


analysis[
    "target"
] = (
    analysis[
        "attempt"
    ]
    .map(
        {
            "A1": 0,
            "A3": 1,
        }
    )
    .astype(int)
)


print(
    f"A1/A3 rows used: {len(analysis)}"
)


print(
    "A1 rows: "
    f"{(analysis['target'] == 0).sum()}"
)


print(
    "A3 rows: "
    f"{(analysis['target'] == 1).sum()}"
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "workout_session",
    "true_subject",
    "exercise",
    "attempt",
    "target",
]


for feature_list in FEATURE_SETS.values():

    required_columns.extend(
        feature_list
    )


required_columns = list(
    dict.fromkeys(
        required_columns
    )
)


missing_columns = [
    column
    for column in required_columns
    if column not in analysis.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


if analysis[
    list(
        dict.fromkeys(
            POSE
            + WEARABLE
            + HR
        )
    )
].isna().any().any():

    raise RuntimeError(
        "\nMissing sensor values remain in matched fusion cohort."
    )


# ============================================================
# STORAGE
# ============================================================

fold_rows = []

prediction_rows = []


# ============================================================
# PER-EXERCISE FUSION EXPERIMENT
# ============================================================

for exercise in sorted(
    analysis[
        "exercise"
    ].unique()
):

    exercise_df = (
        analysis.loc[
            analysis[
                "exercise"
            ] == exercise
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    y = (
        exercise_df[
            "target"
        ]
        .to_numpy()
    )


    groups = (
        exercise_df[
            "true_subject"
        ]
        .to_numpy()
    )


    unique_subjects = np.unique(
        groups
    )


    print(
        "\n"
        + "-" * 88
    )

    print(
        f"{exercise}"
    )

    print(
        f"Rows: {len(exercise_df)} | "
        f"Trajectories: {len(exercise_df) // 2} | "
        f"Subjects: {len(unique_subjects)}"
    )


    logo = LeaveOneGroupOut()


    for model_name, features in FEATURE_SETS.items():

        X = (
            exercise_df[
                features
            ]
            .to_numpy()
        )


        out_of_fold_predictions = np.full(
            len(
                exercise_df
            ),
            np.nan,
        )


        out_of_fold_probabilities = np.full(
            len(
                exercise_df
            ),
            np.nan,
        )


        for (
            fold_number,
            (
                train_index,
                test_index,
            )
        ) in enumerate(
            logo.split(
                X,
                y,
                groups,
            ),
            start=1,
        ):

            X_train = X[
                train_index
            ]

            X_test = X[
                test_index
            ]

            y_train = y[
                train_index
            ]

            y_test = y[
                test_index
            ]


            train_subjects = groups[
                train_index
            ]

            test_subjects = groups[
                test_index
            ]


            held_out_subject = (
                np.unique(
                    test_subjects
                )
            )


            if len(
                held_out_subject
            ) != 1:

                raise RuntimeError(
                    "\nLOSO fold contained more than one "
                    "held-out subject."
                )


            held_out_subject = int(
                held_out_subject[0]
            )


            # ------------------------------------------------
            # SUBJECT-BALANCED SAMPLE WEIGHTS
            # ------------------------------------------------
            #
            # Some true subjects completed multiple workout
            # sessions.
            #
            # Give each training subject approximately equal
            # total weight so subjects with multiple sessions
            # do not dominate the fitted model.

            train_subject_series = pd.Series(
                train_subjects
            )


            subject_counts = (
                train_subject_series
                .value_counts()
                .to_dict()
            )


            sample_weights = np.array(
                [
                    1.0
                    / subject_counts[
                        subject
                    ]
                    for subject in train_subjects
                ],
                dtype=float,
            )


            # Normalise mean weight to 1.
            sample_weights = (
                sample_weights
                / sample_weights.mean()
            )


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            model = RandomForestClassifier(
                **RF_PARAMETERS
            )


            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights,
            )


            predictions = (
                model.predict(
                    X_test
                )
            )


            probabilities = (
                model.predict_proba(
                    X_test
                )[
                    :,
                    1
                ]
            )


            out_of_fold_predictions[
                test_index
            ] = predictions


            out_of_fold_probabilities[
                test_index
            ] = probabilities


            # Each complete held-out subject contributes both
            # A1 and A3, so balanced accuracy and macro F1 are
            # valid within each fold.

            fold_accuracy = accuracy_score(
                y_test,
                predictions,
            )


            fold_balanced_accuracy = (
                balanced_accuracy_score(
                    y_test,
                    predictions,
                )
            )


            fold_macro_f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )


            try:

                fold_auc = roc_auc_score(
                    y_test,
                    probabilities,
                )

            except ValueError:

                fold_auc = np.nan


            fold_rows.append(
                {
                    "exercise":
                        exercise,

                    "model":
                        model_name,

                    "fold":
                        fold_number,

                    "held_out_true_subject":
                        held_out_subject,

                    "n_train_rows":
                        len(
                            train_index
                        ),

                    "n_test_rows":
                        len(
                            test_index
                        ),

                    "accuracy":
                        fold_accuracy,

                    "balanced_accuracy":
                        fold_balanced_accuracy,

                    "macro_f1":
                        fold_macro_f1,

                    "roc_auc":
                        fold_auc,
                }
            )


        # ----------------------------------------------------
        # SAVE OUT-OF-FOLD PREDICTIONS
        # ----------------------------------------------------

        if (
            np.isnan(
                out_of_fold_predictions
            )
            .any()
        ):

            raise RuntimeError(
                "\nMissing out-of-fold predictions."
            )


        if (
            np.isnan(
                out_of_fold_probabilities
            )
            .any()
        ):

            raise RuntimeError(
                "\nMissing out-of-fold probabilities."
            )


        for row_index in range(
            len(
                exercise_df
            )
        ):

            prediction_rows.append(
                {
                    "exercise":
                        exercise,

                    "model":
                        model_name,

                    "workout_session":
                        exercise_df.loc[
                            row_index,
                            "workout_session",
                        ],

                    "true_subject":
                        int(
                            exercise_df.loc[
                                row_index,
                                "true_subject",
                            ]
                        ),

                    "attempt":
                        exercise_df.loc[
                            row_index,
                            "attempt",
                        ],

                    "true_target":
                        int(
                            y[
                                row_index
                            ]
                        ),

                    "predicted_target":
                        int(
                            out_of_fold_predictions[
                                row_index
                            ]
                        ),

                    "probability_A3":
                        float(
                            out_of_fold_probabilities[
                                row_index
                            ]
                        ),
                }
            )


# ============================================================
# SAVE FOLD RESULTS + PREDICTIONS
# ============================================================

fold_results = pd.DataFrame(
    fold_rows
)


predictions = pd.DataFrame(
    prediction_rows
)


fold_results.to_csv(
    TABLES_OUT
    / "07_fusion_loso_fold_results.csv",
    index=False,
)


predictions.to_csv(
    DATA_OUT
    / "07_fusion_out_of_fold_predictions.csv",
    index=False,
)


# ============================================================
# POOLED OUT-OF-FOLD PERFORMANCE
# ============================================================

pooled_rows = []


for (
    exercise,
    model_name
), group in predictions.groupby(
    [
        "exercise",
        "model",
    ]
):

    y_true = (
        group[
            "true_target"
        ]
        .to_numpy()
    )


    y_pred = (
        group[
            "predicted_target"
        ]
        .to_numpy()
    )


    y_prob = (
        group[
            "probability_A3"
        ]
        .to_numpy()
    )


    pooled_rows.append(
        {
            "exercise":
                exercise,

            "model":
                model_name,

            "n_rows":
                len(
                    group
                ),

            "n_true_subjects":
                group[
                    "true_subject"
                ].nunique(),

            "accuracy":
                accuracy_score(
                    y_true,
                    y_pred,
                ),

            "balanced_accuracy":
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                ),

            "macro_f1":
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),

            "roc_auc":
                roc_auc_score(
                    y_true,
                    y_prob,
                ),
        }
    )


pooled_results = pd.DataFrame(
    pooled_rows
)


pooled_results.to_csv(
    TABLES_OUT
    / "07_fusion_pooled_oof_results.csv",
    index=False,
)


# ============================================================
# FOLD DISTRIBUTION SUMMARY
# ============================================================

fold_summary = (
    fold_results
    .groupby(
        [
            "exercise",
            "model",
        ],
        as_index=False,
    )
    .agg(

        n_folds=(
            "fold",
            "size",
        ),

        mean_fold_balanced_accuracy=(
            "balanced_accuracy",
            "mean",
        ),

        sd_fold_balanced_accuracy=(
            "balanced_accuracy",
            "std",
        ),

        mean_fold_macro_f1=(
            "macro_f1",
            "mean",
        ),

        sd_fold_macro_f1=(
            "macro_f1",
            "std",
        ),

        mean_fold_roc_auc=(
            "roc_auc",
            "mean",
        ),

        sd_fold_roc_auc=(
            "roc_auc",
            "std",
        ),
    )
)


fold_summary.to_csv(
    TABLES_OUT
    / "07_fusion_fold_summary.csv",
    index=False,
)


# ============================================================
# FUSION GAIN
# ============================================================

fusion_gain_rows = []


for exercise in sorted(
    pooled_results[
        "exercise"
    ].unique()
):

    exercise_results = (
        pooled_results.loc[
            pooled_results[
                "exercise"
            ] == exercise
        ]
        .set_index(
            "model"
        )
    )


    for metric in [
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
    ]:

        pose = (
            exercise_results.loc[
                "Pose",
                metric,
            ]
        )

        wearable = (
            exercise_results.loc[
                "Wearable",
                metric,
            ]
        )

        hr = (
            exercise_results.loc[
                "HR",
                metric,
            ]
        )

        pose_watch = (
            exercise_results.loc[
                "Pose+Wearable",
                metric,
            ]
        )

        pose_hr = (
            exercise_results.loc[
                "Pose+HR",
                metric,
            ]
        )

        wearable_hr = (
            exercise_results.loc[
                "Wearable+HR",
                metric,
            ]
        )

        all_modalities = (
            exercise_results.loc[
                "Pose+Wearable+HR",
                metric,
            ]
        )


        best_single = max(
            pose,
            wearable,
            hr,
        )


        best_two_modality = max(
            pose_watch,
            pose_hr,
            wearable_hr,
        )


        fusion_gain_rows.append(
            {
                "exercise":
                    exercise,

                "metric":
                    metric,

                "Pose":
                    pose,

                "Wearable":
                    wearable,

                "HR":
                    hr,

                "Pose+Wearable":
                    pose_watch,

                "Pose+HR":
                    pose_hr,

                "Wearable+HR":
                    wearable_hr,

                "Pose+Wearable+HR":
                    all_modalities,

                "best_single_modality":
                    best_single,

                "best_two_modality":
                    best_two_modality,

                "Pose+Wearable_gain_over_best_pose_or_watch":
                    pose_watch
                    - max(
                        pose,
                        wearable,
                    ),

                "All_gain_over_best_single":
                    all_modalities
                    - best_single,

                "All_gain_over_best_two_modality":
                    all_modalities
                    - best_two_modality,

                "HR_increment_over_Pose+Wearable":
                    all_modalities
                    - pose_watch,
            }
        )


fusion_gain = pd.DataFrame(
    fusion_gain_rows
)


fusion_gain.to_csv(
    TABLES_OUT
    / "07_fusion_gain_summary.csv",
    index=False,
)


# ============================================================
# OVERALL ACROSS-EXERCISE DESCRIPTIVE SUMMARY
# ============================================================

overall_model_summary = (
    pooled_results
    .groupby(
        "model",
        as_index=False,
    )
    .agg(

        n_exercises=(
            "exercise",
            "nunique",
        ),

        median_balanced_accuracy=(
            "balanced_accuracy",
            "median",
        ),

        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean",
        ),

        median_macro_f1=(
            "macro_f1",
            "median",
        ),

        mean_macro_f1=(
            "macro_f1",
            "mean",
        ),

        median_roc_auc=(
            "roc_auc",
            "median",
        ),

        mean_roc_auc=(
            "roc_auc",
            "mean",
        ),
    )
)


overall_model_summary.to_csv(
    TABLES_OUT
    / "07_fusion_overall_model_summary.csv",
    index=False,
)


# ============================================================
# METHOD RECORD
# ============================================================

method_record = """
SCRIPT 07 — MULTIMODAL FUSION

Prediction task
---------------
Binary discrimination of A1 versus A3 within each exercise.

A1 = first repeated attempt.
A3 = third repeated attempt.

A2 is excluded from this binary fusion experiment.

The target is repeated-performance stage, NOT fatigue.

Feature sets
------------
Pose: 4 features.
Wearable: 4 features.
HR: 1 feature.
Pose + Wearable: 8 features.
Pose + HR: 5 features.
Wearable + HR: 5 features.
Pose + Wearable + HR: 9 features.

Matched cohort
--------------
Only HR-complete trajectories are used so all feature sets are evaluated
on identical observations.

154 complete trajectories.
462 A1/A2/A3 rows originally.
308 A1/A3 rows used for classification.
9 true subjects overall.

Exercise-specific models
------------------------
A separate model is trained for each exercise.

Exercise identity is therefore not available as a shortcut for predicting
attempt stage.

Validation
----------
Leave-One-True-Subject-Out cross-validation.

Every workout session belonging to the held-out true subject is excluded
from training.

This prevents subject/session leakage and evaluates generalisation to
unseen participants.

Repeated sessions
-----------------
Some true subjects completed multiple workout sessions.

Training observations are sample-weighted so that each true subject
contributes approximately equal total training weight.

Classifier
----------
Random Forest.

Fixed settings:
n_estimators = 500
random_state = 42
n_jobs = -1
class_weight = balanced_subsample
max_features = sqrt

No hyperparameter optimisation is performed.

Metrics
-------
Primary:
Balanced accuracy
Macro F1

Supporting:
ROC-AUC
Accuracy

Pooled out-of-fold predictions are used for primary exercise/model metrics.

Fold-level metrics are also retained to show variability across held-out
subjects.

Fusion gain
-----------
Fusion gain is calculated as a direct difference in predictive performance.

No arbitrary fusion or sensitivity score is created.

Interpretation
--------------
If a fused feature set performs better than the relevant individual
modalities under the same LOSO folds, this supports the conclusion that
the combined modalities provide complementary information for
discriminating early versus late repeated attempts.

It does NOT demonstrate fatigue detection.
"""


(
    LOGS_OUT
    / "07_method_record.txt"
).write_text(
    method_record.strip(),
    encoding="utf-8",
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n" + "-" * 88)
print("OVERALL MODEL SUMMARY ACROSS EXERCISES")
print("-" * 88)


print(
    overall_model_summary
    .sort_values(
        "median_balanced_accuracy",
        ascending=False,
    )
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 88)
print("PER-EXERCISE BALANCED ACCURACY")
print("-" * 88)


balanced_accuracy_table = (
    pooled_results
    .pivot(
        index="exercise",
        columns="model",
        values="balanced_accuracy",
    )
)


print(
    balanced_accuracy_table.to_string(
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 88)
print("FUSION GAIN — BALANCED ACCURACY")
print("-" * 88)


ba_gain = (
    fusion_gain.loc[
        fusion_gain[
            "metric"
        ] == "balanced_accuracy"
    ]
)


print(
    ba_gain[
        [
            "exercise",
            "best_single_modality",
            "Pose+Wearable",
            "Pose+Wearable+HR",
            "Pose+Wearable_gain_over_best_pose_or_watch",
            "All_gain_over_best_single",
            "HR_increment_over_Pose+Wearable",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "=" * 88)
print("SCRIPT 07 COMPLETE")
print("=" * 88)


print(
    "\nPrimary results saved to:"
)


print(
    TABLES_OUT
    / "07_fusion_pooled_oof_results.csv"
)


print(
    "\nFusion gain saved to:"
)


print(
    TABLES_OUT
    / "07_fusion_gain_summary.csv"
)


print(
    "\nThese models discriminate A1 from A3."
)


print(
    "They do NOT classify fatigue."
)   