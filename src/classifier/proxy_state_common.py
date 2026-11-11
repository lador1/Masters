from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

MMFIT_ROOT = Path(r"C:\Masters")
DATA_DIR = MMFIT_ROOT / "data" / "features" / "mmfit"
RESULTS_DIR = MMFIT_ROOT / "results" / "mmfit"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_directories():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def detect_join_key(left_df, right_df, preferred=None):
    if preferred is None:
        preferred = [
            ["participant", "start_time", "end_time"],
            ["participant", "squat_segment_id"],
            ["participant", "squat_attempt_id"],
            ["participant", "squat_segment_number"],
            ["participant", "squat_attempt_number"],
            ["participant"],
        ]

    left_cols = set(left_df.columns)
    right_cols = set(right_df.columns)
    for candidate in preferred:
        if set(candidate).issubset(left_cols) and set(candidate).issubset(right_cols):
            return candidate

    common = [col for col in left_df.columns if col in right_df.columns]
    if not common:
        raise ValueError(
            "No common columns found between datasets. "
            f"Left columns: {list(left_df.columns[:10])}; Right columns: {list(right_df.columns[:10])}"
        )

    fallbacks = [
        "participant",
        "squat_segment_id",
        "squat_attempt_id",
        "squat_segment_number",
        "squat_attempt_number",
        "start_time",
        "end_time",
    ]

    for col in fallbacks:
        if col in common:
            return [col]

    return [common[0]]


def feature_group_columns(all_columns):
    all_columns = [str(col) for col in all_columns]
    pose = [
        col for col in all_columns
        if (col.startswith("pose_") or col.startswith("joint_"))
        and col not in {"pose_valid", "pose_frames", "pose_joint_count"}
    ]
    wearable = [
        col for col in all_columns
        if col.startswith(("sw_l_acc_", "sw_r_acc_", "sw_l_gyr_", "sw_r_gyr_"))
    ]
    hr = [
        col for col in all_columns
        if col.startswith(("sw_l_hr_", "sw_r_hr_", "hr_"))
        and col not in {"hr_quality_flag", "hr_disagreement_flag", "hr_left_available", "hr_right_available", "hr_any_available", "hr_both_available"}
    ]
    additional = [
        col for col in all_columns
        if col in {"hr_mean_combined", "hr_std_combined", "hr_min_combined", "hr_max_combined", "hr_range_combined", "hr_change_combined", "hr_slope_combined"}
    ]
    hr = list(dict.fromkeys(hr + additional))
    return {
        "pose": pose,
        "wearable": wearable,
        "hr": hr,
    }


def select_feature_columns(df):
    candidate_cols = [str(col) for col in df.columns]
    feature_map = feature_group_columns(candidate_cols)
    feature_groups = {col: "pose" for col in feature_map["pose"]}
    feature_groups.update({col: "wearable" for col in feature_map["wearable"]})
    feature_groups.update({col: "hr" for col in feature_map["hr"]})
    return feature_groups


def load_required_csv(path, required_cols, dataset_name):
    df = pd.read_csv(path)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    return df


def safe_classification_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def compute_confusion_and_report(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    precision = []
    recall = []
    for label in labels:
        tp = cm_df.loc[label, label]
        predicted_total = cm_df.loc[:, label].sum()
        actual_total = cm_df.loc[label, :].sum()
        precision_value = (tp / predicted_total) if predicted_total else 0.0
        recall_value = (tp / actual_total) if actual_total else 0.0
        precision.append(precision_value)
        recall.append(recall_value)
    f1_scores = []
    for p, r in zip(precision, recall):
        f1_scores.append((2 * p * r / (p + r)) if (p + r) else 0.0)
    report_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1_scores,
            "support": [cm_df.loc[label, :].sum() for label in labels],
        },
        index=labels,
    )
    report_df.index.name = "proxy_state"
    return cm_df, report_df


def save_confusion_figure(cm_df, output_path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_df.values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(cm_df.columns)))
    ax.set_yticks(np.arange(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns)
    ax.set_yticklabels(cm_df.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, int(cm_df.iloc[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_feature_importance_figure(df, output_path, title="Top feature importances"):
    top = df.head(20).copy()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"].astype(str)[::-1], top["importance"].astype(float)[::-1], color="steelblue")
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_bar_chart(values, labels, output_path, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c72b0", "#dd8452", "#55a868"][: len(labels)]
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=20)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def print_class_warning(labels):
    counts = pd.Series(labels).value_counts()
    if (counts < 2).any():
        print("WARNING: Insufficient samples for reliable multiclass classification.")
        print(counts)
