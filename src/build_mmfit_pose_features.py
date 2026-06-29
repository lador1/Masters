from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")
OUT_DIR = Path(r"C:\Masters\data\features\mmfit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

label_files = sorted(RAW_DIR.rglob("*_labels.csv"))


def basic_pose_features(pose_3d, start_frame, end_frame):
    """
    pose_3d shape from MM-Fit looks like:
    (3, frames, joints)

    Axis 0 = x, y, z coordinates
    Axis 1 = time/frame
    Axis 2 = body joints
    """

    total_frames = pose_3d.shape[1]

    start_frame = int(start_frame)
    end_frame = int(end_frame)

    # Keep frame range safe
    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    if end_frame <= start_frame:
        return {
            "pose_valid": 0,
            "pose_frames": 0,
            "pose_movement_mean": np.nan,
            "pose_movement_std": np.nan,
            "pose_range_mean": np.nan,
            "pose_range_max": np.nan,
            "pose_stability_proxy": np.nan,
        }

    # segment shape: coordinates x frames x joints
    segment = pose_3d[:, start_frame:end_frame, :]

    # Movement between consecutive frames
    diffs = np.diff(segment, axis=1)

    # Euclidean movement magnitude per joint per frame
    movement = np.sqrt(np.sum(diffs ** 2, axis=0))

    # Range of motion per joint across segment
    coord_range = np.nanmax(segment, axis=1) - np.nanmin(segment, axis=1)
    joint_range = np.sqrt(np.sum(coord_range ** 2, axis=0))

    # Simple stability proxy:
    # lower average centre movement = more stable
    centre = np.nanmean(segment, axis=2)
    centre_diffs = np.diff(centre, axis=1)
    centre_movement = np.sqrt(np.sum(centre_diffs ** 2, axis=0))

    return {
        "pose_valid": 1,
        "pose_frames": int(segment.shape[1]),
        "pose_movement_mean": float(np.nanmean(movement)),
        "pose_movement_std": float(np.nanstd(movement)),
        "pose_range_mean": float(np.nanmean(joint_range)),
        "pose_range_max": float(np.nanmax(joint_range)),
        "pose_stability_proxy": float(np.nanmean(centre_movement)),
    }


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

        row.update(basic_pose_features(pose_3d, start_time, end_time))

        rows.append(row)


features = pd.DataFrame(rows)

output_path = OUT_DIR / "mmfit_pose_features_basic.csv"
features.to_csv(output_path, index=False)

print("Pose feature table saved to:")
print(output_path)

print("\nShape:")
print(features.shape)

print("\nPreview:")
print(features.head(20))

print("\nMissing values:")
print(features.isna().sum())

print("\nExercise counts:")
print(features["exercise"].value_counts())

print("\nPose valid counts:")
print(features["pose_valid"].value_counts())