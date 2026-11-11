from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# PATHS
# ============================================================

FEATURE_DIR = Path(r"C:\Masters\data\features\mmfit")

POSE_PATH = FEATURE_DIR / "mmfit_pose_features_v2_fixed_clean.csv"
WEARABLE_PATH = FEATURE_DIR / "mmfit_wearable_features_basic.csv"

OUT_DIR = Path(
    r"C:\Masters\results\final_analysis\exercise_recognition_clean"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# MM-FIT WORKOUT -> SUBJECT MAPPING
#
# From the official MM-Fit EDA notebook.
# This is important because w00-w20 are workout/session IDs,
# not 21 separate participants.
# ============================================================

WORKOUT_TO_SUBJECT = {
    "w00": 2,
    "w01": 0,
    "w02": 1,
    "w03": 0,
    "w04": 1,
    "w05": 2,
    "w06": 0,
    "w07": 1,
    "w08": 0,
    "w09": 1,
    "w10": 0,
    "w11": 1,
    "w12": 3,
    "w13": 4,
    "w14": 0,
    "w15": 1,
    "w16": 5,
    "w17": 6,
    "w18": 7,
    "w19": 8,
    "w20": 9,
}

# Official MM-Fit Human3.6M 3D joint ordering.
JOINT_NAMES = {
    0: "hip_root",
    1: "left_hip",
    2: "left_knee",
    3: "left_foot",
    4: "right_hip",
    5: "right_knee",
    6: "right_foot",
    7: "spine",
    8: "thorax",
    9: "neck_nose",
    10: "head",
    11: "right_shoulder",
    12: "right_elbow",
    13: "right_wrist",
    14: "left_shoulder",
    15: "left_elbow",
    16: "left_wrist",
}

RANDOM_STATE = 42
N_TREES = 500

# ============================================================
# FEATURE DEFINITIONS
# ============================================================
#
# POSE:
# Exactly 17 joints x 4 simple segment-level movement summaries:
#   1. range
#   2. mean frame-to-frame movement
#   3. SD of frame-to-frame movement
#   4. maximum frame-to-frame movement
#
# We deliberately DO NOT use:
#   pose_frames
#   duration
#   repetition count
#   pose validity flags
#   exercise IDs / session IDs
#   the large advanced squat-specific feature bank
#
# WATCH:
# 2 wrists x 2 sensors x 10 simple inertial summaries = 40.
# Heart rate is excluded from this exercise-recognition validation.
# ============================================================

POSE_STATS = [
    "range",
    "speed_mean",
    "speed_std",
    "speed_max",
]

POSE_FEATURES = [
    f"joint_{joint:02d}_{stat}"
    for joint in range(17)
    for stat in POSE_STATS
]

WATCH_PREFIXES = [
    "sw_l_acc",
    "sw_r_acc",
    "sw_l_gyr",
    "sw_r_gyr",
]

WATCH_STATS = [
    "mag_mean",
    "mag_std",
    "mag_min",
    "mag_max",
    "x_mean",
    "y_mean",
    "z_mean",
    "x_std",
    "y_std",
    "z_std",
]

WATCH_FEATURES = [
    f"{prefix}_{stat}"
    for prefix in WATCH_PREFIXES
    for stat in WATCH_STATS
]

MERGE_COLS = [
    "participant",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
]

# ============================================================
# HELPERS
# ============================================================

def readable_feature_name(feature):
    """
    Make feature names easier to read in the saved importance table.
    """

    if feature.startswith("joint_"):
        parts = feature.split("_")
        joint_index = int(parts[1])
        stat = "_".join(parts[2:])

        # The original extractor named these columns "speed", but it
        # calculated frame-to-frame displacement without dividing by dt.
        # Use accurate readable labels in the result tables.
        stat = {
            "speed_mean": "movement_mean",
            "speed_std": "movement_std",
            "speed_max": "movement_max",
        }.get(stat, stat)

        return f"{JOINT_NAMES[joint_index]}_{stat}"

    replacements = {
        "sw_l_acc": "left_watch_acc",
        "sw_r_acc": "right_watch_acc",
        "sw_l_gyr": "left_watch_gyro",
        "sw_r_gyr": "right_watch_gyro",
    }

    readable = feature

    for old, new in replacements.items():
        readable = readable.replace(old, new)

    return readable


def check_required_columns(df, columns, table_name):
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"\nMissing columns in {table_name}:\n"
            + "\n".join(missing)
        )


def build_model():
    """
    Same Random Forest settings are used for pose, watch and fusion.
    """

    return RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )


def save_confusion_matrix(y_true, y_pred, labels, model_name):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(11, 9))

    image = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted exercise")
    ax.set_ylabel("True exercise")
    ax.set_title(
        f"{model_name} Random Forest\n"
        "Leave-one-subject-out pooled confusion matrix"
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    path = OUT_DIR / f"{model_name.lower()}_confusion_matrix.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CLEAN MM-FIT EXERCISE-RECOGNITION VALIDATION")
print("=" * 70)

pose = pd.read_csv(POSE_PATH)
wearable = pd.read_csv(WEARABLE_PATH)

print("\nPose table:")
print(POSE_PATH)
print("Shape:", pose.shape)

print("\nWearable table:")
print(WEARABLE_PATH)
print("Shape:", wearable.shape)

check_required_columns(
    pose,
    MERGE_COLS + POSE_FEATURES,
    "pose table",
)

check_required_columns(
    wearable,
    MERGE_COLS + WATCH_FEATURES,
    "wearable table",
)

# ============================================================
# MERGE SO ALL THREE MODELS USE EXACTLY THE SAME EXERCISE SETS
# ============================================================

data = pd.merge(
    pose[MERGE_COLS + POSE_FEATURES],
    wearable[MERGE_COLS + WATCH_FEATURES],
    on=MERGE_COLS,
    how="inner",
)

before_dropna = len(data)

data = data.dropna(
    subset=POSE_FEATURES + WATCH_FEATURES
).reset_index(drop=True)

print("\nMerged rows before missing-value removal:", before_dropna)
print("Rows used by ALL three models:", len(data))

# ============================================================
# ADD TRUE SUBJECT ID
# ============================================================

data["subject_id"] = data["participant"].map(WORKOUT_TO_SUBJECT)

if data["subject_id"].isna().any():
    unmapped = sorted(
        data.loc[
            data["subject_id"].isna(),
            "participant",
        ].unique()
    )

    raise ValueError(
        "These workout IDs are missing from WORKOUT_TO_SUBJECT: "
        + ", ".join(unmapped)
    )

data["subject_id"] = data["subject_id"].astype(int)

print("\nWorkout sessions:", data["participant"].nunique())
print("True subjects:", data["subject_id"].nunique())

print("\nRows by subject:")
print(
    data.groupby("subject_id")
    .size()
    .rename("rows")
    .to_string()
)

print("\nExercise counts:")
print(data["exercise"].value_counts().to_string())

# Save the exact common analysis dataset.
analysis_dataset_path = OUT_DIR / "exercise_recognition_common_dataset.csv"
data.to_csv(analysis_dataset_path, index=False)

# ============================================================
# MODEL FEATURE SETS
# ============================================================

FEATURE_SETS = {
    "Pose": POSE_FEATURES,
    "Watch": WATCH_FEATURES,
    "Fusion": POSE_FEATURES + WATCH_FEATURES,
}

print("\nFeature counts:")
for name, features in FEATURE_SETS.items():
    print(f"{name}: {len(features)}")

# ============================================================
# LEAVE-ONE-SUBJECT-OUT VALIDATION
# ============================================================

y = data["exercise"].astype(str).to_numpy()
groups = data["subject_id"].to_numpy()

class_labels = sorted(data["exercise"].astype(str).unique())

logo = LeaveOneGroupOut()

fold_rows = []
prediction_rows = []
importance_rows = []

for fold_number, (train_idx, test_idx) in enumerate(
    logo.split(data, y, groups),
    start=1,
):
    held_out_subjects = sorted(
        data.iloc[test_idx]["subject_id"].unique()
    )

    if len(held_out_subjects) != 1:
        raise RuntimeError(
            "LeaveOneGroupOut produced more than one held-out subject."
        )

    held_out_subject = int(held_out_subjects[0])

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("\n" + "-" * 70)
    print(
        f"FOLD {fold_number}: "
        f"Held-out subject {held_out_subject}"
    )
    print(
        f"Train rows: {len(train_idx)} | "
        f"Test rows: {len(test_idx)}"
    )

    for model_name, feature_columns in FEATURE_SETS.items():
        X_train = (
            data.iloc[train_idx][feature_columns]
            .to_numpy(dtype=float)
        )

        X_test = (
            data.iloc[test_idx][feature_columns]
            .to_numpy(dtype=float)
        )

        model = build_model()
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, pred)

        macro_f1 = f1_score(
            y_test,
            pred,
            average="macro",
            labels=class_labels,
            zero_division=0,
        )

        print(
            f"{model_name:<7} "
            f"Accuracy = {accuracy:.4f} | "
            f"Macro F1 = {macro_f1:.4f}"
        )

        fold_rows.append({
            "fold": fold_number,
            "held_out_subject": held_out_subject,
            "model": model_name,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        test_metadata = data.iloc[test_idx][
            [
                "participant",
                "subject_id",
                "start_time",
                "end_time",
                "exercise",
            ]
        ].reset_index(drop=True)

        for row_index in range(len(test_metadata)):
            prediction_rows.append({
                "fold": fold_number,
                "model": model_name,
                "workout_id": test_metadata.loc[
                    row_index,
                    "participant",
                ],
                "subject_id": int(
                    test_metadata.loc[
                        row_index,
                        "subject_id",
                    ]
                ),
                "start_time": test_metadata.loc[
                    row_index,
                    "start_time",
                ],
                "end_time": test_metadata.loc[
                    row_index,
                    "end_time",
                ],
                "true_exercise": y_test[row_index],
                "predicted_exercise": pred[row_index],
            })

        for feature, importance in zip(
            feature_columns,
            model.feature_importances_,
        ):
            importance_rows.append({
                "fold": fold_number,
                "held_out_subject": held_out_subject,
                "model": model_name,
                "feature": feature,
                "readable_feature": readable_feature_name(
                    feature
                ),
                "importance": float(importance),
            })

# ============================================================
# SAVE FOLD RESULTS
# ============================================================

fold_results = pd.DataFrame(fold_rows)
predictions = pd.DataFrame(prediction_rows)
importance = pd.DataFrame(importance_rows)

fold_results_path = OUT_DIR / "rf_loso_fold_results.csv"
predictions_path = OUT_DIR / "rf_loso_predictions.csv"

fold_results.to_csv(fold_results_path, index=False)
predictions.to_csv(predictions_path, index=False)

# ============================================================
# POOLED RESULTS
#
# Every held-out prediction from all 10 subject folds is combined.
# ============================================================

overall_rows = []

for model_name in FEATURE_SETS:
    model_predictions = predictions[
        predictions["model"] == model_name
    ]

    y_true = model_predictions[
        "true_exercise"
    ].to_numpy()

    y_pred = model_predictions[
        "predicted_exercise"
    ].to_numpy()

    pooled_accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    pooled_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=class_labels,
        zero_division=0,
    )

    model_folds = fold_results[
        fold_results["model"] == model_name
    ]

    overall_rows.append({
        "model": model_name,
        "n_features": len(FEATURE_SETS[model_name]),
        "n_rows": len(y_true),
        "n_subjects": data["subject_id"].nunique(),
        "pooled_accuracy": pooled_accuracy,
        "pooled_macro_f1": pooled_macro_f1,
        "fold_accuracy_mean": model_folds[
            "accuracy"
        ].mean(),
        "fold_accuracy_std": model_folds[
            "accuracy"
        ].std(ddof=1),
        "fold_macro_f1_mean": model_folds[
            "macro_f1"
        ].mean(),
        "fold_macro_f1_std": model_folds[
            "macro_f1"
        ].std(ddof=1),
    })

    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        zero_division=0,
        output_dict=True,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )

    report_df.to_csv(
        OUT_DIR
        / f"{model_name.lower()}_classification_report.csv",
        index=False,
    )

    save_confusion_matrix(
        y_true,
        y_pred,
        class_labels,
        model_name,
    )

overall = pd.DataFrame(overall_rows)

overall_path = OUT_DIR / "rf_loso_overall_results.csv"
overall.to_csv(overall_path, index=False)

# ============================================================
# FEATURE IMPORTANCE ACROSS FOLDS
#
# Importance is calculated separately in every held-out-subject fold.
# Final table reports mean and SD across folds.
# ============================================================

importance_summary = (
    importance.groupby(
        [
            "model",
            "feature",
            "readable_feature",
        ],
        as_index=False,
    )
    .agg(
        importance_mean=("importance", "mean"),
        importance_std=("importance", "std"),
    )
)

importance_summary["importance_rank"] = (
    importance_summary.groupby("model")[
        "importance_mean"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)

importance_summary = importance_summary.sort_values(
    [
        "model",
        "importance_rank",
    ]
)

importance_path = OUT_DIR / "rf_loso_feature_importance.csv"
importance_summary.to_csv(importance_path, index=False)

# Save compact top-20 importance tables.
for model_name in FEATURE_SETS:
    top = (
        importance_summary[
            importance_summary["model"] == model_name
        ]
        .head(20)
        .copy()
    )

    top.to_csv(
        OUT_DIR
        / f"{model_name.lower()}_top20_feature_importance.csv",
        index=False,
    )

# ============================================================
# SIMPLE MODEL COMPARISON FIGURE
# ============================================================

plot_data = overall.set_index("model").loc[
    ["Pose", "Watch", "Fusion"]
]

x = np.arange(len(plot_data))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(
    x - width / 2,
    plot_data["pooled_accuracy"],
    width,
    label="Accuracy",
)

bars2 = ax.bar(
    x + width / 2,
    plot_data["pooled_macro_f1"],
    width,
    label="Macro F1",
)

ax.set_xticks(x)
ax.set_xticklabels(plot_data.index)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title(
    "MM-Fit exercise recognition\n"
    "Leave-one-subject-out validation"
)
ax.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

fig.tight_layout()
comparison_figure_path = (
    OUT_DIR / "rf_loso_model_comparison.png"
)
fig.savefig(
    comparison_figure_path,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL POOLED LOSO RESULTS")
print("=" * 70)

display_cols = [
    "model",
    "n_features",
    "n_rows",
    "n_subjects",
    "pooled_accuracy",
    "pooled_macro_f1",
    "fold_accuracy_mean",
    "fold_accuracy_std",
    "fold_macro_f1_mean",
    "fold_macro_f1_std",
]

print(
    overall[display_cols]
    .round(4)
    .to_string(index=False)
)

print("\nTop 10 pose features:")
print(
    importance_summary[
        importance_summary["model"] == "Pose"
    ][
        [
            "importance_rank",
            "readable_feature",
            "importance_mean",
            "importance_std",
        ]
    ]
    .head(10)
    .round(5)
    .to_string(index=False)
)

print("\nTop 10 watch features:")
print(
    importance_summary[
        importance_summary["model"] == "Watch"
    ][
        [
            "importance_rank",
            "readable_feature",
            "importance_mean",
            "importance_std",
        ]
    ]
    .head(10)
    .round(5)
    .to_string(index=False)
)

print("\nSaved outputs:")
print(overall_path)
print(fold_results_path)
print(predictions_path)
print(importance_path)
print(comparison_figure_path)

print("\nDone.")
