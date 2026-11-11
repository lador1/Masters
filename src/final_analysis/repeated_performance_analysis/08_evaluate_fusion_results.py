"""
08_evaluate_fusion_results.py

PRIMARY THESIS ANALYSIS — FUSION EVALUATION AND VISUALISATION

Purpose
-------
Evaluate and visualise the frozen A1-vs-A3 multimodal fusion experiment
produced by Script 07.

NO models are retrained in this script.

NO features are changed.

NO hyperparameter tuning is performed.

NO alternative algorithms are tested.

The purpose is to report the existing experiment transparently.

This script produces:
1. Per-exercise balanced-accuracy heatmap.
2. Per-exercise Macro-F1 heatmap.
3. Per-exercise ROC-AUC heatmap.
4. Fusion-gain figure.
5. Across-exercise performance-distribution figure.
6. Pooled out-of-fold confusion summaries.
7. Fold-variability tables.
8. A clear table showing where fusion numerically improved,
   tied, or reduced performance.

IMPORTANT
---------
A1 and A3 are chronological repeated-attempt labels.

A1 = first occurrence of the exercise.
A3 = third occurrence of the exercise.

A3 is NOT defined as fatigue.

Therefore classification performance represents discrimination of
repeated-performance stage, not fatigue detection.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ============================================================
# PATHS
# ============================================================

RESULTS_ROOT = Path(
    r"C:\Masters\results\final_analysis\repeated_performance_analysis"
)

TABLES_OUT = RESULTS_ROOT / "tables"
FIGURES_OUT = RESULTS_ROOT / "figures"
LOGS_OUT = RESULTS_ROOT / "logs"
DATA_OUT = RESULTS_ROOT / "data"


POOLED_INPUT = (
    TABLES_OUT
    / "07_fusion_pooled_oof_results.csv"
)

FOLD_INPUT = (
    TABLES_OUT
    / "07_fusion_loso_fold_results.csv"
)

GAIN_INPUT = (
    TABLES_OUT
    / "07_fusion_gain_summary.csv"
)

PREDICTION_INPUT = (
    DATA_OUT
    / "07_fusion_out_of_fold_predictions.csv"
)


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
# LOAD
# ============================================================

required_files = [
    POOLED_INPUT,
    FOLD_INPUT,
    GAIN_INPUT,
    PREDICTION_INPUT,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired Script 07 output not found:\n{path}"
        )


pooled = pd.read_csv(
    POOLED_INPUT
)

folds = pd.read_csv(
    FOLD_INPUT
)

gain = pd.read_csv(
    GAIN_INPUT
)

predictions = pd.read_csv(
    PREDICTION_INPUT
)


print("\n" + "=" * 88)
print("MM-FIT REPEATED PERFORMANCE ANALYSIS")
print("SCRIPT 08 — FUSION EVALUATION AND VISUALISATION")
print("=" * 88)


print(
    f"\nPooled result rows: {len(pooled)}"
)

print(
    f"Fold result rows: {len(folds)}"
)

print(
    f"Prediction rows: {len(predictions)}"
)

print(
    f"Exercises: {pooled['exercise'].nunique()}"
)

print(
    f"Models: {pooled['model'].nunique()}"
)


# ============================================================
# FIXED MODEL ORDER
# ============================================================

MODEL_ORDER = [
    "Pose",
    "Wearable",
    "HR",
    "Pose+Wearable",
    "Pose+HR",
    "Wearable+HR",
    "Pose+Wearable+HR",
]


missing_models = (
    set(MODEL_ORDER)
    - set(
        pooled[
            "model"
        ].unique()
    )
)


if missing_models:

    raise RuntimeError(
        "\nMissing expected models:\n"
        + "\n".join(
            sorted(
                missing_models
            )
        )
    )


# ============================================================
# RECOMPUTE OVERALL DESCRIPTIVE SUMMARY
# ============================================================

overall_summary = (
    pooled
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


overall_summary[
    "model_order"
] = (
    overall_summary[
        "model"
    ]
    .map(
        {
            model: i
            for i, model
            in enumerate(
                MODEL_ORDER
            )
        }
    )
)


overall_summary = (
    overall_summary
    .sort_values(
        "model_order"
    )
    .drop(
        columns=[
            "model_order"
        ]
    )
)


overall_summary.to_csv(
    TABLES_OUT
    / "08_overall_model_summary.csv",
    index=False,
)


# ============================================================
# EXERCISE x MODEL PERFORMANCE TABLE
# ============================================================

performance_table = pooled[
    [
        "exercise",
        "model",
        "n_rows",
        "n_true_subjects",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
    ]
].copy()


performance_table.to_csv(
    TABLES_OUT
    / "08_exercise_model_performance.csv",
    index=False,
)


# ============================================================
# NUMERICALLY HIGHEST MODEL PER EXERCISE
# ============================================================

# This simply reports the maximum observed balanced accuracy.
#
# It is NOT evidence that the model is statistically superior.

highest_rows = []


for exercise, group in pooled.groupby(
    "exercise"
):

    maximum = (
        group[
            "balanced_accuracy"
        ]
        .max()
    )


    matching = (
        group.loc[
            np.isclose(
                group[
                    "balanced_accuracy"
                ],
                maximum,
                atol=1e-12,
            )
        ]
        .copy()
    )


    highest_rows.append(
        {
            "exercise":
                exercise,

            "highest_observed_balanced_accuracy":
                maximum,

            "model_or_models":
                " | ".join(
                    sorted(
                        matching[
                            "model"
                        ].tolist()
                    )
                ),

            "n_tied_models":
                len(
                    matching
                ),
        }
    )


highest_observed = pd.DataFrame(
    highest_rows
)


highest_observed.to_csv(
    TABLES_OUT
    / "08_highest_observed_model_by_exercise.csv",
    index=False,
)


# ============================================================
# FUSION GAIN — BALANCED ACCURACY
# ============================================================

ba_gain = (
    gain.loc[
        gain[
            "metric"
        ] == "balanced_accuracy"
    ]
    .copy()
)


def classify_difference(
    value
):

    if np.isclose(
        value,
        0,
        atol=1e-12,
    ):

        return "tie"

    elif value > 0:

        return "numerical_improvement"

    else:

        return "numerical_reduction"


ba_gain[
    "pose_watch_result"
] = (
    ba_gain[
        "Pose+Wearable_gain_over_best_pose_or_watch"
    ]
    .apply(
        classify_difference
    )
)


ba_gain[
    "all_modalities_result"
] = (
    ba_gain[
        "All_gain_over_best_single"
    ]
    .apply(
        classify_difference
    )
)


ba_gain[
    "hr_increment_result"
] = (
    ba_gain[
        "HR_increment_over_Pose+Wearable"
    ]
    .apply(
        classify_difference
    )
)


ba_gain.to_csv(
    TABLES_OUT
    / "08_balanced_accuracy_fusion_gain.csv",
    index=False,
)


# ============================================================
# SUMMARY COUNTS OF FUSION GAIN
# ============================================================

fusion_count_rows = []


for comparison_name, column in [

    (
        "Pose+Wearable vs best Pose/Watch",
        "pose_watch_result",
    ),

    (
        "All modalities vs best single modality",
        "all_modalities_result",
    ),

    (
        "Adding HR to Pose+Wearable",
        "hr_increment_result",
    ),
]:

    counts = (
        ba_gain[
            column
        ]
        .value_counts()
    )


    fusion_count_rows.append(
        {
            "comparison":
                comparison_name,

            "numerical_improvement":
                int(
                    counts.get(
                        "numerical_improvement",
                        0,
                    )
                ),

            "tie":
                int(
                    counts.get(
                        "tie",
                        0,
                    )
                ),

            "numerical_reduction":
                int(
                    counts.get(
                        "numerical_reduction",
                        0,
                    )
                ),
        }
    )


fusion_counts = pd.DataFrame(
    fusion_count_rows
)


fusion_counts.to_csv(
    TABLES_OUT
    / "08_fusion_gain_counts.csv",
    index=False,
)


# ============================================================
# LOSO FOLD VARIABILITY
# ============================================================

fold_variability = (
    folds
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

        mean_balanced_accuracy=(
            "balanced_accuracy",
            "mean",
        ),

        sd_balanced_accuracy=(
            "balanced_accuracy",
            "std",
        ),

        minimum_balanced_accuracy=(
            "balanced_accuracy",
            "min",
        ),

        maximum_balanced_accuracy=(
            "balanced_accuracy",
            "max",
        ),

        mean_macro_f1=(
            "macro_f1",
            "mean",
        ),

        sd_macro_f1=(
            "macro_f1",
            "std",
        ),

        mean_roc_auc=(
            "roc_auc",
            "mean",
        ),

        sd_roc_auc=(
            "roc_auc",
            "std",
        ),
    )
)


fold_variability.to_csv(
    TABLES_OUT
    / "08_loso_fold_variability.csv",
    index=False,
)


# ============================================================
# POOLED CONFUSION SUMMARY
# ============================================================

# These combine the out-of-fold predictions from all nine
# exercise-specific models for each feature set.
#
# This is an aggregate error summary.
# It is NOT the performance of one global classifier.

confusion_rows = []


for model_name in MODEL_ORDER:

    model_predictions = (
        predictions.loc[
            predictions[
                "model"
            ] == model_name
        ]
        .copy()
    )


    matrix = confusion_matrix(
        model_predictions[
            "true_target"
        ],
        model_predictions[
            "predicted_target"
        ],
        labels=[
            0,
            1,
        ],
    )


    tn, fp, fn, tp = (
        matrix.ravel()
    )


    confusion_rows.append(
        {
            "model":
                model_name,

            "true_A1_predicted_A1":
                int(
                    tn
                ),

            "true_A1_predicted_A3":
                int(
                    fp
                ),

            "true_A3_predicted_A1":
                int(
                    fn
                ),

            "true_A3_predicted_A3":
                int(
                    tp
                ),

            "total_predictions":
                int(
                    matrix.sum()
                ),
        }
    )


confusion_summary = pd.DataFrame(
    confusion_rows
)


confusion_summary.to_csv(
    TABLES_OUT
    / "08_pooled_confusion_summary.csv",
    index=False,
)


# ============================================================
# HEATMAP FUNCTION
# ============================================================

def create_metric_heatmap(
    metric,
    title,
    filename,
):

    matrix = (
        pooled
        .pivot(
            index="exercise",
            columns="model",
            values=metric,
        )
        .reindex(
            columns=MODEL_ORDER
        )
        .sort_index()
    )


    values = (
        matrix
        .to_numpy(
            dtype=float
        )
    )


    fig, ax = plt.subplots(
        figsize=(
            13,
            8,
        )
    )


    image = ax.imshow(
        values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )


    ax.set_xticks(
        range(
            len(
                matrix.columns
            )
        )
    )


    ax.set_xticklabels(
        matrix.columns,
        rotation=35,
        ha="right",
    )


    ax.set_yticks(
        range(
            len(
                matrix.index
            )
        )
    )


    ax.set_yticklabels(
        [
            exercise.replace(
                "_",
                " ",
            )
            for exercise in matrix.index
        ]
    )


    ax.set_title(
        title
    )


    for row_index in range(
        len(
            matrix.index
        )
    ):

        for column_index in range(
            len(
                matrix.columns
            )
        ):

            value = values[
                row_index,
                column_index
            ]


            if not np.isnan(
                value
            ):

                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
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


    plt.close(
        fig
    )


    return output_path


# ============================================================
# FIGURES 1–3 — PERFORMANCE HEATMAPS
# ============================================================

ba_heatmap = create_metric_heatmap(
    "balanced_accuracy",
    "A1 vs A3 Subject-Independent Balanced Accuracy",
    "08_fusion_balanced_accuracy_heatmap.png",
)


f1_heatmap = create_metric_heatmap(
    "macro_f1",
    "A1 vs A3 Subject-Independent Macro F1",
    "08_fusion_macro_f1_heatmap.png",
)


auc_heatmap = create_metric_heatmap(
    "roc_auc",
    "A1 vs A3 Subject-Independent ROC-AUC",
    "08_fusion_roc_auc_heatmap.png",
)


# ============================================================
# FIGURE 4 — FUSION GAIN
# ============================================================

ba_gain_plot = (
    ba_gain
    .sort_values(
        "exercise"
    )
    .reset_index(
        drop=True
    )
)


x = np.arange(
    len(
        ba_gain_plot
    )
)


width = 0.27


fig, ax = plt.subplots(
    figsize=(
        14,
        7,
    )
)


ax.bar(
    x - width,
    ba_gain_plot[
        "Pose+Wearable_gain_over_best_pose_or_watch"
    ],
    width,
    label="Pose+Wearable vs best Pose/Watch",
)


ax.bar(
    x,
    ba_gain_plot[
        "All_gain_over_best_single"
    ],
    width,
    label="All modalities vs best single",
)


ax.bar(
    x + width,
    ba_gain_plot[
        "HR_increment_over_Pose+Wearable"
    ],
    width,
    label="HR added to Pose+Wearable",
)


ax.axhline(
    0,
    linewidth=1,
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
        for exercise in ba_gain_plot[
            "exercise"
        ]
    ],
    rotation=35,
    ha="right",
)


ax.set_ylabel(
    "Balanced-accuracy difference"
)


ax.set_xlabel(
    "Exercise"
)


ax.set_title(
    "Observed Fusion Gain by Exercise"
)


ax.legend()


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "08_fusion_gain_balanced_accuracy.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# FIGURE 5 — ACROSS-EXERCISE PERFORMANCE DISTRIBUTION
# ============================================================

box_data = []


for model_name in MODEL_ORDER:

    values = (
        pooled.loc[
            pooled[
                "model"
            ] == model_name,
            "balanced_accuracy",
        ]
        .to_numpy()
    )


    box_data.append(
        values
    )


fig, ax = plt.subplots(
    figsize=(
        13,
        7,
    )
)


ax.boxplot(
    box_data,
    tick_labels=MODEL_ORDER,
)


ax.axhline(
    0.5,
    linewidth=1,
)


ax.set_ylabel(
    "Balanced accuracy"
)


ax.set_xlabel(
    "Feature set"
)


ax.set_title(
    "Distribution of A1-vs-A3 Performance Across Exercises"
)


ax.tick_params(
    axis="x",
    rotation=35,
)


fig.tight_layout()


fig.savefig(
    FIGURES_OUT
    / "08_balanced_accuracy_distribution.png",
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ============================================================
# FIGURE 6+ — AGGREGATE CONFUSION MATRIX PER MODEL
# ============================================================

for model_name in MODEL_ORDER:

    model_predictions = (
        predictions.loc[
            predictions[
                "model"
            ] == model_name
        ]
    )


    matrix = confusion_matrix(
        model_predictions[
            "true_target"
        ],
        model_predictions[
            "predicted_target"
        ],
        labels=[
            0,
            1,
        ],
    )


    fig, ax = plt.subplots(
        figsize=(
            6,
            5,
        )
    )


    image = ax.imshow(
        matrix,
        aspect="auto",
    )


    ax.set_xticks(
        [
            0,
            1,
        ]
    )


    ax.set_xticklabels(
        [
            "Predicted A1",
            "Predicted A3",
        ]
    )


    ax.set_yticks(
        [
            0,
            1,
        ]
    )


    ax.set_yticklabels(
        [
            "True A1",
            "True A3",
        ]
    )


    for row_index in range(
        2
    ):

        for column_index in range(
            2
        ):

            ax.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index
                    ]
                ),
                ha="center",
                va="center",
            )


    ax.set_title(
        f"{model_name}: Aggregate Out-of-Fold Confusion Matrix"
    )


    fig.colorbar(
        image,
        ax=ax,
        shrink=0.8,
    )


    fig.tight_layout()


    safe_name = (
        model_name
        .replace(
            "+",
            "_plus_",
        )
        .replace(
            " ",
            "_",
        )
    )


    fig.savefig(
        FIGURES_OUT
        / f"08_confusion_{safe_name}.png",
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# FIGURE MANIFEST
# ============================================================

figure_manifest = pd.DataFrame(
    [
        {
            "figure":
                "08_fusion_balanced_accuracy_heatmap.png",

            "purpose":
                "Exercise-specific balanced accuracy for all seven feature sets.",
        },

        {
            "figure":
                "08_fusion_macro_f1_heatmap.png",

            "purpose":
                "Exercise-specific Macro F1 for all seven feature sets.",
        },

        {
            "figure":
                "08_fusion_roc_auc_heatmap.png",

            "purpose":
                "Exercise-specific ROC-AUC for all seven feature sets.",
        },

        {
            "figure":
                "08_fusion_gain_balanced_accuracy.png",

            "purpose":
                "Shows whether selected fusion combinations numerically improved or reduced balanced accuracy.",
        },

        {
            "figure":
                "08_balanced_accuracy_distribution.png",

            "purpose":
                "Shows the distribution of balanced accuracy across the nine exercises for each feature set.",
        },
    ]
)


figure_manifest.to_csv(
    TABLES_OUT
    / "08_figure_manifest.csv",
    index=False,
)


# ============================================================
# METHOD / EXPLAINABILITY RECORD
# ============================================================

method_record = """
SCRIPT 08 — FUSION EVALUATION AND VISUALISATION

Purpose
-------
Evaluate the frozen Script 07 multimodal classification experiment.

No models were retrained.

No hyperparameters were changed.

No feature selection was performed.

No additional classifiers were tested.

Classification target
---------------------
A1 = chronologically first exercise attempt.
A3 = chronologically third exercise attempt.

These labels are determined entirely by exercise order.

Sensor measurements, medians, effect sizes and heart-rate values do
not determine whether a row is labelled A1 or A3.

Therefore the classification experiment is not circular with respect
to the sensor features.

A3 is NOT assumed to represent fatigue.

The classification task tests whether sensor measurements contain
information associated with repeated-performance stage.

Validation
----------
All performance values originate from Leave-One-True-Subject-Out
cross-validation performed in Script 07.

Every prediction therefore comes from a model that was not trained
using any workout session belonging to that test subject.

Metrics
-------
Balanced accuracy:
Average of sensitivity to A1 and sensitivity to A3.

Macro F1:
F1 score calculated separately for A1 and A3 and then averaged.

ROC-AUC:
Measures ranking/discrimination using predicted A3 probabilities.

Balanced accuracy and Macro F1 are treated as the primary metrics.

Fusion gain
-----------
Fusion gain is a simple arithmetic difference between model performance.

Examples:

Pose+Wearable gain =
Pose+Wearable balanced accuracy
minus
the higher balanced accuracy of Pose and Wearable individually.

All-modality gain =
Pose+Wearable+HR balanced accuracy
minus
the strongest individual modality.

A positive number means the fused feature set achieved a numerically
higher score.

A negative number means it achieved a numerically lower score.

A zero indicates an observed tie.

These differences are descriptive and do not establish statistical
superiority.

Interpretation
--------------
The experiment evaluates whether combined sensor measurements improve
subject-independent discrimination of early versus late repeated
exercise attempts.

It does NOT evaluate fatigue classification.

It does NOT establish clinical deterioration.

It does NOT establish that a modality is universally superior.

Because only 7-9 true subjects are available per exercise within the
HR-complete cohort, results are interpreted as proof-of-concept
subject-independent evidence.

Aggregate confusion matrices
----------------------------
Confusion matrices combine out-of-fold predictions from the separate
exercise-specific models.

They provide an overall error summary but do not represent one global
classifier trained across all exercises.
"""


(
    LOGS_OUT
    / "08_method_and_explainability_record.txt"
).write_text(
    method_record.strip(),
    encoding="utf-8",
)


# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("\n" + "-" * 88)
print("OVERALL MODEL PERFORMANCE")
print("-" * 88)


print(
    overall_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 88)
print("HIGHEST OBSERVED BALANCED ACCURACY BY EXERCISE")
print("-" * 88)


print(
    highest_observed.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 88)
print("FUSION GAIN COUNTS")
print("-" * 88)


print(
    fusion_counts.to_string(
        index=False
    )
)


print("\n" + "-" * 88)
print("BALANCED-ACCURACY FUSION GAIN")
print("-" * 88)


print(
    ba_gain[
        [
            "exercise",
            "Pose+Wearable_gain_over_best_pose_or_watch",
            "pose_watch_result",
            "All_gain_over_best_single",
            "all_modalities_result",
            "HR_increment_over_Pose+Wearable",
            "hr_increment_result",
        ]
    ]
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n" + "-" * 88)
print("AGGREGATE OUT-OF-FOLD CONFUSION COUNTS")
print("-" * 88)


print(
    confusion_summary.to_string(
        index=False
    )
)


print("\n" + "=" * 88)
print("SCRIPT 08 COMPLETE")
print("=" * 88)


print(
    "\nNo models were retrained or modified."
)


print(
    "\nThe Script 07 experiment is now fully evaluated "
    "and documented."
)


print(
    "\nFigures saved to:"
)


print(
    FIGURES_OUT
)