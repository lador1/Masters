from pathlib import Path
import pandas as pd
import re

RESULTS_DIR = Path(r"C:\Masters\results\mmfit")

INPUT_PATH = RESULTS_DIR / "pose_baseline_v2_fixed_feature_importance.csv"
OUTPUT_PATH = RESULTS_DIR / "pose_baseline_v2_fixed_feature_importance_named.csv"

importance = pd.read_csv(INPUT_PATH)

# After removing pose column 0, our joint_00 to joint_16 correspond to the remaining pose joints.
# This mapping assumes the remaining 17 joints follow common COCO keypoint order.
# Keep this assumption documented until the exact MM-Fit joint-name list is confirmed.
JOINT_NAME_MAP = {
    "joint_00": "nose",
    "joint_01": "left_eye",
    "joint_02": "right_eye",
    "joint_03": "left_ear",
    "joint_04": "right_ear",
    "joint_05": "left_shoulder",
    "joint_06": "right_shoulder",
    "joint_07": "left_elbow",
    "joint_08": "right_elbow",
    "joint_09": "left_wrist",
    "joint_10": "right_wrist",
    "joint_11": "left_hip",
    "joint_12": "right_hip",
    "joint_13": "left_knee",
    "joint_14": "right_knee",
    "joint_15": "left_ankle",
    "joint_16": "right_ankle",
}

def rename_feature(feature_name: str) -> str:
    for joint_code, joint_name in JOINT_NAME_MAP.items():
        if feature_name.startswith(joint_code + "_"):
            return feature_name.replace(joint_code, joint_name, 1)
    return feature_name

importance["feature_named"] = importance["feature"].apply(rename_feature)

# Put readable name first
importance = importance[["feature", "feature_named", "importance"]]

importance.to_csv(OUTPUT_PATH, index=False)

print("Saved named feature importance to:")
print(OUTPUT_PATH)

print("\nTop 30 named features:")
print(importance.head(30))