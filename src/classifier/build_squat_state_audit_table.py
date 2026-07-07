from pathlib import Path
import pandas as pd

INPUT_PATH = Path(r"C:\Masters\results\mmfit\squat_metrics_by_participant.csv")
OUT_DIR = Path(r"C:\Masters\results\mmfit")
OUT_PATH = OUT_DIR / "squat_state_audit_table.csv"

data = pd.read_csv(INPUT_PATH)

# -------------------------------------------------------
# 1. These are the exact raw squat metrics used
# -------------------------------------------------------

raw_metrics = [
    "avg_pose_movement",
    "avg_pose_range",
    "avg_pose_stability",
    "avg_watch_acc_mean",
    "avg_watch_acc_variability",
    "avg_watch_gyr_mean",
    "avg_watch_gyr_variability",
]

# -------------------------------------------------------
# 2. Convert each raw metric to a z-score
# -------------------------------------------------------
# z-score = (participant value - group average) / group standard deviation
#
# Meaning:
# 0 = average
# positive = above group average
# negative = below group average

def z_score(series):
    return (series - series.mean()) / series.std()

data["pose_movement_score"] = z_score(data["avg_pose_movement"])
data["pose_range_score"] = z_score(data["avg_pose_range"])

# IMPORTANT:
# avg_pose_stability behaves like movement variation.
# Higher raw value = more variation / less stable.
# So we multiply by -1 so that higher score = more stable.
data["pose_stability_score"] = -z_score(data["avg_pose_stability"])

data["watch_acc_score"] = z_score(data["avg_watch_acc_mean"])
data["watch_acc_variability_score"] = z_score(data["avg_watch_acc_variability"])
data["watch_gyr_score"] = z_score(data["avg_watch_gyr_mean"])
data["watch_gyr_variability_score"] = z_score(data["avg_watch_gyr_variability"])

# -------------------------------------------------------
# 3. Combined movement scores
# -------------------------------------------------------

# Movement strength =
# average of body movement, body range, watch acceleration, watch rotation
data["movement_strength_score"] = data[
    [
        "pose_movement_score",
        "pose_range_score",
        "watch_acc_score",
        "watch_gyr_score",
    ]
].mean(axis=1)

# Movement variability =
# average of watch acceleration variability and watch gyroscope variability
data["movement_variability_score"] = data[
    [
        "watch_acc_variability_score",
        "watch_gyr_variability_score",
    ]
].mean(axis=1)

# Engagement-like =
# stronger movement + more stable movement
data["engagement_like_score"] = data[
    [
        "movement_strength_score",
        "pose_stability_score",
    ]
].mean(axis=1)

# Fatigue-like =
# low movement strength
data["fatigue_like_score"] = -data["movement_strength_score"]

# Frustration-like =
# high movement variability
data["frustration_like_score"] = data["movement_variability_score"]

# -------------------------------------------------------
# 4. Assign state and explain exactly why
# -------------------------------------------------------

def assign_state_and_reason(row):
    if row["fatigue_like_score"] > 0.75:
        return pd.Series({
            "derived_state": "fatigue_like",
            "state_reason": (
                f"fatigue_like_score = {row['fatigue_like_score']:.4f}, "
                f"which is above 0.75. This means movement strength was lower than average."
            ),
            "rule_triggered": "fatigue_like_score > 0.75",
            "trigger_score": row["fatigue_like_score"],
        })

    elif row["frustration_like_score"] > 0.75:
        return pd.Series({
            "derived_state": "frustration_like",
            "state_reason": (
                f"frustration_like_score = {row['frustration_like_score']:.4f}, "
                f"which is above 0.75. This means movement variability was higher than average."
            ),
            "rule_triggered": "frustration_like_score > 0.75",
            "trigger_score": row["frustration_like_score"],
        })

    elif row["engagement_like_score"] > 0.5:
        return pd.Series({
            "derived_state": "engaged_like",
            "state_reason": (
                f"engagement_like_score = {row['engagement_like_score']:.4f}, "
                f"which is above 0.5. This means movement was stronger and/or more stable than average."
            ),
            "rule_triggered": "engagement_like_score > 0.5",
            "trigger_score": row["engagement_like_score"],
        })

    else:
        return pd.Series({
            "derived_state": "neutral_like",
            "state_reason": (
                "No prototype threshold was passed. "
                "Movement strength, variability, and engagement-like scores were not high enough "
                "to trigger fatigue_like, frustration_like, or engaged_like."
            ),
            "rule_triggered": "no threshold passed",
            "trigger_score": 0,
        })

state_info = data.apply(assign_state_and_reason, axis=1)
data = pd.concat([data, state_info], axis=1)

# -------------------------------------------------------
# 5. Add plain-English metric explanations
# -------------------------------------------------------

data["movement_strength_formula"] = (
    "(pose_movement_score + pose_range_score + watch_acc_score + watch_gyr_score) / 4"
)

data["movement_variability_formula"] = (
    "(watch_acc_variability_score + watch_gyr_variability_score) / 2"
)

data["engagement_like_formula"] = (
    "(movement_strength_score + pose_stability_score) / 2"
)

data["fatigue_like_formula"] = (
    "-movement_strength_score"
)

data["frustration_like_formula"] = (
    "movement_variability_score"
)

# -------------------------------------------------------
# 6. Reorder columns so it is easy to inspect
# -------------------------------------------------------

ordered_cols = [
    "participant",

    # raw metrics
    "squat_segments",
    "avg_reps",
    "avg_duration",
    "avg_pose_movement",
    "avg_pose_range",
    "avg_pose_stability",
    "avg_watch_acc_mean",
    "avg_watch_acc_variability",
    "avg_watch_gyr_mean",
    "avg_watch_gyr_variability",

    # z scores
    "pose_movement_score",
    "pose_range_score",
    "pose_stability_score",
    "watch_acc_score",
    "watch_acc_variability_score",
    "watch_gyr_score",
    "watch_gyr_variability_score",

    # combined scores
    "movement_strength_score",
    "movement_variability_score",
    "engagement_like_score",
    "fatigue_like_score",
    "frustration_like_score",

    # final label and explanation
    "derived_state",
    "rule_triggered",
    "trigger_score",
    "state_reason",

    # formulas
    "movement_strength_formula",
    "movement_variability_formula",
    "engagement_like_formula",
    "fatigue_like_formula",
    "frustration_like_formula",
]

audit = data[ordered_cols].round(4)

audit.to_csv(OUT_PATH, index=False)

print("Saved audit table to:")
print(OUT_PATH)

print("\nState counts:")
print(audit["derived_state"].value_counts())

print("\nAudit table preview:")
print(
    audit[
        [
            "participant",
            "movement_strength_score",
            "movement_variability_score",
            "engagement_like_score",
            "fatigue_like_score",
            "frustration_like_score",
            "derived_state",
            "rule_triggered",
            "state_reason",
        ]
    ]
)