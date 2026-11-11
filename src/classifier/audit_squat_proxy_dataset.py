from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from proxy_state_common import DATA_DIR, FIGURES_DIR, RESULTS_DIR, detect_join_key, ensure_directories, print_class_warning

FEATURE_PATH = DATA_DIR / "mmfit_fusion_features_with_hr_cleaned.csv"
SEGMENT_PATH = RESULTS_DIR / "squat_metrics_by_segment_usable_only.csv"
LABEL_PATH = RESULTS_DIR / "squat_state_scores_per_participant_exercise_attempt.csv"

SUMMARY_PATH = RESULTS_DIR / "proxy_state_audit_summary.csv"
COUNTS_PATH = RESULTS_DIR / "proxy_state_counts.csv"
COUNTS_BY_PARTICIPANT_PATH = RESULTS_DIR / "proxy_state_counts_by_participant.csv"
FIGURE_PATH = FIGURES_DIR / "proxy_state_counts.png"


def print_section(title):
    print(f"\n--- {title} ---")


ensure_directories()

feature_df = pd.read_csv(FEATURE_PATH)
segment_df = pd.read_csv(SEGMENT_PATH)
label_df = pd.read_csv(LABEL_PATH)

print_section("Dataset shapes")
print("feature_df shape:", feature_df.shape)
print("segment_df shape:", segment_df.shape)
print("label_df shape:", label_df.shape)

print_section("Column names")
print("feature columns sample:", feature_df.columns[:10].tolist(), "... total:", len(feature_df.columns))
print("segment columns sample:", segment_df.columns[:10].tolist(), "... total:", len(segment_df.columns))
print("label columns:", label_df.columns.tolist())

print_section("First 5 rows")
print(feature_df.head().to_string(index=False))
print(segment_df.head().to_string(index=False))
print(label_df.head().to_string(index=False))

print_section("Missing values")
print("feature missing counts:\n", feature_df.isna().sum().sort_values(ascending=False).head(15).to_string())
print("segment missing counts:\n", segment_df.isna().sum().sort_values(ascending=False).head(15).to_string())
print("label missing counts:\n", label_df.isna().sum().sort_values(ascending=False).head(15).to_string())

print_section("Duplicate attempt IDs")
feature_duplicates = feature_df.duplicated(subset=["participant", "start_time", "end_time"]).sum()
segment_duplicates = segment_df["squat_segment_id"].duplicated().sum()
label_duplicates = label_df["squat_attempt_id"].duplicated().sum()
print("feature duplicate rows by participant/start/end:", feature_duplicates)
print("segment duplicate squat_segment_id:", segment_duplicates)
print("label duplicate squat_attempt_id:", label_duplicates)

print_section("Participant counts")
print("feature participants:", feature_df["participant"].nunique())
print("segment participants:", segment_df["participant"].nunique())
print("label participants:", label_df["participant"].nunique())
print("attempts per participant:\n", label_df["participant"].value_counts().sort_index().to_string())

print_section("Proxy-state counts")
state_counts = label_df["proxy_state"].value_counts(dropna=False)
print(state_counts.to_string())
print("proxy_state percentage:\n", (state_counts / state_counts.sum() * 100).round(2).to_string())
print_class_warning(label_df["proxy_state"])

join_feature_segment = detect_join_key(feature_df, segment_df)
print_section("Join key audit")
print("Using feature -> segment join keys:", join_feature_segment)

summary_rows = [
    {"metric": "feature_rows", "value": int(feature_df.shape[0])},
    {"metric": "segment_rows", "value": int(segment_df.shape[0])},
    {"metric": "label_rows", "value": int(label_df.shape[0])},
    {"metric": "feature_participants", "value": int(feature_df["participant"].nunique())},
    {"metric": "segment_participants", "value": int(segment_df["participant"].nunique())},
    {"metric": "label_participants", "value": int(label_df["participant"].nunique())},
    {"metric": "feature_columns", "value": int(feature_df.shape[1])},
    {"metric": "segment_columns", "value": int(segment_df.shape[1])},
    {"metric": "label_columns", "value": int(label_df.shape[1])},
    {"metric": "proxy_state_classes", "value": int(label_df["proxy_state"].nunique())},
    {"metric": "duplicate_feature_attempt_rows", "value": int(feature_duplicates)},
    {"metric": "duplicate_segment_ids", "value": int(segment_duplicates)},
    {"metric": "duplicate_label_attempt_ids", "value": int(label_duplicates)},
]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_PATH, index=False)

counts_df = state_counts.reset_index()
counts_df.columns = ["proxy_state", "count"]
counts_df["percentage"] = counts_df["count"] / counts_df["count"].sum() * 100.0
counts_df.to_csv(COUNTS_PATH, index=False)

counts_by_participant = (
    label_df.groupby(["participant", "proxy_state"], dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values(["participant", "proxy_state"])
)
counts_by_participant.to_csv(COUNTS_BY_PARTICIPANT_PATH, index=False)

fig, ax = plt.subplots(figsize=(8, 5))
counts_df.set_index("proxy_state")["count"].plot(kind="bar", ax=ax, color="steelblue")
ax.set_title("Proxy-state class counts")
ax.set_xlabel("Proxy state")
ax.set_ylabel("Count")
fig.tight_layout()
fig.savefig(FIGURE_PATH)
plt.close(fig)

print("\nSaved audit outputs:")
print(SUMMARY_PATH)
print(COUNTS_PATH)
print(COUNTS_BY_PARTICIPANT_PATH)
print(FIGURE_PATH)
