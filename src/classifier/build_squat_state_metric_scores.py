from pathlib import Path
import pandas as pd

INPUT_PATH = Path(r"C:\Masters\results\mmfit\squat_metrics_by_participant.csv")
OUT_DIR = Path(r"C:\Masters\results\mmfit")
OUT_PATH = OUT_DIR / "squat_state_metric_scores.csv"

data = pd.read_csv(INPUT_PATH)

# Helper: convert a column into a simple relative score
# Above average = positive
# Below average = negative
def z_score(series):
    return (series - series.mean()) / series.std()

# Movement/body scores
data["pose_movement_score"] = z_score(data["avg_pose_movement"])
data["pose_range_score"] = z_score(data["avg_pose_range"])

# Stability score
# Higher pose_stability_proxy means more movement variation.
# For "stable movement", lower variation is better, so we invert it.
data["pose_stability_score"] = -z_score(data["avg_pose_stability"])

# Watch movement scores
data["watch_acc_score"] = z_score(data["avg_watch_acc_mean"])
data["watch_acc_variability_score"] = z_score(data["avg_watch_acc_variability"])
data["watch_gyr_score"] = z_score(data["avg_watch_gyr_mean"])
data["watch_gyr_variability_score"] = z_score(data["avg_watch_gyr_variability"])

# Simple combined scores
data["movement_strength_score"] = data[
    [
        "pose_movement_score",
        "pose_range_score",
        "watch_acc_score",
        "watch_gyr_score",
    ]
].mean(axis=1)

data["movement_variability_score"] = data[
    [
        "watch_acc_variability_score",
        "watch_gyr_variability_score",
    ]
].mean(axis=1)

# Early rough state-style scores
# These are not real emotions yet.
# They are movement-based indicators.
data["engagement_like_score"] = data[
    [
        "movement_strength_score",
        "pose_stability_score",
    ]
].mean(axis=1)

data["fatigue_like_score"] = -data["movement_strength_score"]

data["frustration_like_score"] = data["movement_variability_score"]

# Simple rule-based label
def assign_state(row):
    if row["fatigue_like_score"] > 0.75:
        return "fatigue_like"
    elif row["frustration_like_score"] > 0.75:
        return "frustration_like"
    elif row["engagement_like_score"] > 0.5:
        return "engaged_like"
    else:
        return "neutral_like"

data["derived_state"] = data.apply(assign_state, axis=1)

# Round for readability
data_rounded = data.round(4)

data_rounded.to_csv(OUT_PATH, index=False)

print("Saved squat state metric scores to:")
print(OUT_PATH)

print("\nDerived state counts:")
print(data_rounded["derived_state"].value_counts())

print("\nParticipant state scores:")
print(
    data_rounded[
        [
            "participant",
            "movement_strength_score",
            "movement_variability_score",
            "engagement_like_score",
            "fatigue_like_score",
            "frustration_like_score",
            "derived_state",
        ]
    ]
)