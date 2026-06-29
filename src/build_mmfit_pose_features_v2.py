from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")
OUT_DIR = Path(r"C:\Masters\data\features\mmfit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

label_files = sorted(RAW_DIR.rglob("*_labels.csv"))


def extract_pose_features_v2_fixed(pose_3d, start_frame, end_frame):
    """
    MM-Fit pose_3d appears to use shape:
    (3, frames, 18)

    Axis 0 = x, y, z coordinates
    Axis 1 = frame/time
    Axis 2 = pose columns

    Important:
    The MM-Fit repo indicates pose column 0 is frame/index-related,
    so actual body joints are taken from columns 1:.
    That leaves 17 body joints.
    """

    total_frames = pose_3d.shape[1]

    start_frame = int(start_frame)
    end_frame = int(end_frame)

    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    if end_frame <= start_frame:
        return {
            "pose_valid": 0,
            "pose_frames": 0,
        }

    # FIXED: remove pose column 0
    segment = pose_3d[:, start_frame:end_frame, 1:]

    if segment.shape[1] < 2:
        return {
            "pose_valid": 0,
            "pose_frames": int(segment.shape[1]),
        }

    # FIXED: get joint count after removing column 0
    n_joints = segment.shape[2]

    features = {
        "pose_valid": 1,
        "pose_frames": int(segment.shape[1]),
        "pose_joint_count": int(n_joints),
    }

    # Frame-to-frame movement
    diffs = np.diff(segment, axis=1)

    # Shape: frames-1 x joints
    joint_speed = np.sqrt(np.sum(diffs ** 2, axis=0))

    # Joint range of motion
    coord_range = np.nanmax(segment, axis=1) - np.nanmin(segment, axis=1)
    joint_range = np.sqrt(np.sum(coord_range ** 2, axis=0))

    # Per-joint features
    for j in range(n_joints):
        features[f"joint_{j:02d}_range"] = float(joint_range[j])
        features[f"joint_{j:02d}_speed_mean"] = float(np.nanmean(joint_speed[:, j]))
        features[f"joint_{j:02d}_speed_std"] = float(np.nanstd(joint_speed[:, j]))
        features[f"joint_{j:02d}_speed_max"] = float(np.nanmax(joint_speed[:, j]))

    # Whole-body summary features
    features["pose_movement_mean"] = float(np.nanmean(joint_speed))
    features["pose_movement_std"] = float(np.nanstd(joint_speed))
    features["pose_movement_max"] = float(np.nanmax(joint_speed))
    features["pose_range_mean"] = float(np.nanmean(joint_range))
    features["pose_range_std"] = float(np.nanstd(joint_range))
    features["pose_range_max"] = float(np.nanmax(joint_range))

    # Centre movement / stability proxy
    centre = np.nanmean(segment, axis=2)
    centre_diffs = np.diff(centre, axis=1)
    centre_movement = np.sqrt(np.sum(centre_diffs ** 2, axis=0))

    features["pose_stability_proxy"] = float(np.nanmean(centre_movement))
    features["pose_stability_std"] = float(np.nanstd(centre_movement))
    features["pose_stability_max"] = float(np.nanmax(centre_movement))

    return features


for label_file in label_files:
    participant = label_file.name.split("_")[0]
    participant_dir = label_file.parent

    pose_path = participant_dir / f"{participant}_pose_3d.npy"

    if not pose_path.exists():
        print(f"Missing pose file for {participant}")
        continue

    pose_3d = np.load(pose_path, allow_pickle=True)

    labels = pd.read_csv(
        label_file,
        header=None,
        names=["start_time", "end_time", "reps", "exercise"]
    )

    for _, label in labels.iterrows():
        start_time = label["start_time"]
        end_time = label["end_time"]

        row = {
            "participant": participant,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "reps": label["reps"],
            "exercise": label["exercise"],
        }

        row.update(extract_pose_features_v2_fixed(pose_3d, start_time, end_time))

        rows.append(row)


features = pd.DataFrame(rows)

output_path = OUT_DIR / "mmfit_pose_features_v2_fixed.csv"
features.to_csv(output_path, index=False)

print("Fixed improved pose feature table saved to:")
print(output_path)

print("\nShape:")
print(features.shape)

print("\nPreview:")
print(features.head())

print("\nMissing values:")
print(features.isna().sum().sort_values(ascending=False).head(20))

print("\nPose valid counts:")
print(features["pose_valid"].value_counts())

print("\nPose joint counts:")
print(features["pose_joint_count"].value_counts(dropna=False))

print("\nExercise counts:")
print(features["exercise"].value_counts())