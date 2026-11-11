from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PARTICIPANT = "w01"

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")

OUT_DIR = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\advanced_feature_checks"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


FPS = 30.0


JOINTS = {
    "hip_root": 0,

    "right_hip": 1,
    "right_knee": 2,
    "right_foot": 3,

    "left_hip": 4,
    "left_knee": 5,
    "left_foot": 6,

    "spine": 7,
    "thorax": 8,
    "neck_nose": 9,
    "head": 10,

    "left_shoulder": 11,
    "left_elbow": 12,
    "left_wrist": 13,

    "right_shoulder": 14,
    "right_elbow": 15,
    "right_wrist": 16,
}


X_AXIS = 0
Z_AXIS = 1
Y_AXIS = 2


def angle_3d(a, b, c):
    ba = a - b
    bc = c - b

    numerator = np.sum(
        ba * bc,
        axis=1
    )

    denominator = (
        np.linalg.norm(ba, axis=1)
        *
        np.linalg.norm(bc, axis=1)
    )

    cosine = np.divide(
        numerator,
        denominator,
        out=np.full(
            numerator.shape,
            np.nan,
            dtype=float
        ),
        where=denominator > 1e-12
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine)
    )


def horizontal_distance(a, b):
    delta_x = a[:, X_AXIS] - b[:, X_AXIS]
    delta_z = a[:, Z_AXIS] - b[:, Z_AXIS]

    return np.sqrt(
        delta_x ** 2
        +
        delta_z ** 2
    )


def distance_3d(a, b):
    return np.linalg.norm(
        a - b,
        axis=1
    )


def describe_signal(name, values):
    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        print(
            f"{name}: no valid values"
        )
        return

    print(
        f"{name:<36}"
        f" min={np.min(values):10.3f}"
        f" max={np.max(values):10.3f}"
        f" range={np.ptp(values):10.3f}"
        f" mean={np.mean(values):10.3f}"
        f" std={np.std(values):10.3f}"
    )


participant_dir = (
    RAW_DIR
    / PARTICIPANT
)

pose_path = (
    participant_dir
    / f"{PARTICIPANT}_pose_3d.npy"
)

label_path = (
    participant_dir
    / f"{PARTICIPANT}_labels.csv"
)


if not pose_path.exists():
    raise FileNotFoundError(
        pose_path
    )

if not label_path.exists():
    raise FileNotFoundError(
        label_path
    )


pose_raw = np.load(
    pose_path,
    allow_pickle=True
)

labels = pd.read_csv(
    label_path,
    header=None,
    names=[
        "start_frame",
        "end_frame",
        "reps",
        "exercise",
    ]
)


print("\n========================================")
print("RAW POSE")
print("========================================")

print(
    "Participant:",
    PARTICIPANT
)

print(
    "Pose path:",
    pose_path
)

print(
    "Pose shape:",
    pose_raw.shape
)

print(
    "Pose dtype:",
    pose_raw.dtype
)


squats = labels[
    labels["exercise"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("squats")
].copy()


if squats.empty:
    raise ValueError(
        f"No squat labels found for {PARTICIPANT}"
    )


squat = squats.iloc[0]

start = int(
    squat["start_frame"]
)

end = int(
    squat["end_frame"]
)


print("\n========================================")
print("FIRST SQUAT ATTEMPT")
print("========================================")

print(
    "Start frame:",
    start
)

print(
    "End frame:",
    end
)

print(
    "Frames:",
    end - start
)

print(
    "Duration seconds:",
    round(
        (end - start) / FPS,
        3
    )
)

print(
    "Labelled reps:",
    squat["reps"]
)


segment = pose_raw[
    :,
    start:end,
    1:
]


print(
    "\nSegment shape after removing frame column:"
)

print(
    segment.shape
)


if segment.shape[0] != 3:
    raise ValueError(
        "Expected three coordinate dimensions."
    )

if segment.shape[2] != 17:
    raise ValueError(
        f"Expected 17 joints, got {segment.shape[2]}"
    )


xyz = np.transpose(
    segment,
    (1, 2, 0)
)

frames = np.arange(
    len(xyz)
)


print("\n========================================")
print("POSE QUALITY")
print("========================================")

total_values = xyz.size

finite_values = np.isfinite(
    xyz
).sum()

print(
    "Finite values:",
    finite_values,
    "/",
    total_values
)

print(
    "Finite percentage:",
    round(
        finite_values
        / total_values
        * 100,
        4
    ),
    "%"
)

print(
    "NaN count:",
    np.isnan(xyz).sum()
)

print(
    "Inf count:",
    np.isinf(xyz).sum()
)


hip_root = xyz[
    :,
    JOINTS["hip_root"],
    :
]

right_hip = xyz[
    :,
    JOINTS["right_hip"],
    :
]

right_knee = xyz[
    :,
    JOINTS["right_knee"],
    :
]

right_foot = xyz[
    :,
    JOINTS["right_foot"],
    :
]

left_hip = xyz[
    :,
    JOINTS["left_hip"],
    :
]

left_knee = xyz[
    :,
    JOINTS["left_knee"],
    :
]

left_foot = xyz[
    :,
    JOINTS["left_foot"],
    :
]

spine = xyz[
    :,
    JOINTS["spine"],
    :
]

thorax = xyz[
    :,
    JOINTS["thorax"],
    :
]

left_shoulder = xyz[
    :,
    JOINTS["left_shoulder"],
    :
]

right_shoulder = xyz[
    :,
    JOINTS["right_shoulder"],
    :
]


print("\n========================================")
print("HIP ROOT CHECK")
print("========================================")

describe_signal(
    "root X",
    hip_root[:, X_AXIS]
)

describe_signal(
    "root Z",
    hip_root[:, Z_AXIS]
)

describe_signal(
    "root Y vertical",
    hip_root[:, Y_AXIS]
)

root_absolute_max = np.nanmax(
    np.abs(hip_root)
)

print(
    "\nMaximum absolute root coordinate:",
    round(
        float(root_absolute_max),
        6
    )
)

if root_absolute_max < 1e-6:
    print(
        "Root is effectively fixed at the origin."
    )
elif root_absolute_max < 5:
    print(
        "Root remains very close to the origin."
    )
else:
    print(
        "Root position changes measurably through the segment."
    )


print("\n========================================")
print("JOINT COORDINATE RANGES")
print("========================================")

important_joints = {
    "right hip": right_hip,
    "right knee": right_knee,
    "right foot": right_foot,

    "left hip": left_hip,
    "left knee": left_knee,
    "left foot": left_foot,

    "spine": spine,
    "thorax": thorax,
}


for joint_name, joint_data in important_joints.items():

    print(
        f"\n--- {joint_name} ---"
    )

    describe_signal(
        "X",
        joint_data[:, X_AXIS]
    )

    describe_signal(
        "Z",
        joint_data[:, Z_AXIS]
    )

    describe_signal(
        "Y vertical",
        joint_data[:, Y_AXIS]
    )


right_knee_angle = angle_3d(
    right_hip,
    right_knee,
    right_foot
)

left_knee_angle = angle_3d(
    left_hip,
    left_knee,
    left_foot
)


print("\n========================================")
print("KNEE ANGLES")
print("========================================")

describe_signal(
    "right knee angle",
    right_knee_angle
)

describe_signal(
    "left knee angle",
    left_knee_angle
)


print(
    "\nRight knee percentiles:"
)

print(
    np.nanpercentile(
        right_knee_angle,
        [
            1,
            5,
            25,
            50,
            75,
            95,
            99
        ]
    )
)


print(
    "\nLeft knee percentiles:"
)

print(
    np.nanpercentile(
        left_knee_angle,
        [
            1,
            5,
            25,
            50,
            75,
            95,
            99
        ]
    )
)


right_hip_angle = angle_3d(
    thorax,
    right_hip,
    right_knee
)

left_hip_angle = angle_3d(
    thorax,
    left_hip,
    left_knee
)


print("\n========================================")
print("HIP ANGLES")
print("========================================")

describe_signal(
    "right hip angle",
    right_hip_angle
)

describe_signal(
    "left hip angle",
    left_hip_angle
)


foot_separation_horizontal = (
    horizontal_distance(
        left_foot,
        right_foot
    )
)

knee_separation_horizontal = (
    horizontal_distance(
        left_knee,
        right_knee
    )
)

hip_separation_horizontal = (
    horizontal_distance(
        left_hip,
        right_hip
    )
)


foot_separation_3d = distance_3d(
    left_foot,
    right_foot
)

knee_separation_3d = distance_3d(
    left_knee,
    right_knee
)

hip_separation_3d = distance_3d(
    left_hip,
    right_hip
)


print("\n========================================")
print("LEFT / RIGHT SEPARATION")
print("========================================")

describe_signal(
    "foot separation horizontal",
    foot_separation_horizontal
)

describe_signal(
    "knee separation horizontal",
    knee_separation_horizontal
)

describe_signal(
    "hip separation horizontal",
    hip_separation_horizontal
)

describe_signal(
    "foot separation 3D",
    foot_separation_3d
)

describe_signal(
    "knee separation 3D",
    knee_separation_3d
)

describe_signal(
    "hip separation 3D",
    hip_separation_3d
)


mean_foot_vertical = (
    left_foot[:, Y_AXIS]
    +
    right_foot[:, Y_AXIS]
) / 2.0


root_to_foot_vertical_distance = np.abs(
    hip_root[:, Y_AXIS]
    -
    mean_foot_vertical
)


print("\n========================================")
print("VERTICAL SQUAT POSTURE PROXY")
print("========================================")

describe_signal(
    "root-to-feet vertical distance",
    root_to_foot_vertical_distance
)


output = pd.DataFrame({
    "frame_in_segment":
        frames,

    "absolute_frame":
        frames + start,

    "root_x":
        hip_root[:, X_AXIS],

    "root_z":
        hip_root[:, Z_AXIS],

    "root_y_vertical":
        hip_root[:, Y_AXIS],

    "right_knee_angle_deg":
        right_knee_angle,

    "left_knee_angle_deg":
        left_knee_angle,

    "right_hip_angle_deg":
        right_hip_angle,

    "left_hip_angle_deg":
        left_hip_angle,

    "foot_separation_horizontal":
        foot_separation_horizontal,

    "knee_separation_horizontal":
        knee_separation_horizontal,

    "hip_separation_horizontal":
        hip_separation_horizontal,

    "foot_separation_3d":
        foot_separation_3d,

    "knee_separation_3d":
        knee_separation_3d,

    "hip_separation_3d":
        hip_separation_3d,

    "root_to_feet_vertical_distance":
        root_to_foot_vertical_distance,
})


csv_path = (
    OUT_DIR
    / (
        f"{PARTICIPANT}"
        "_first_squat_corrected_pose_check.csv"
    )
)

output.to_csv(
    csv_path,
    index=False
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    frames,
    left_knee_angle,
    label="Left knee"
)

plt.plot(
    frames,
    right_knee_angle,
    label="Right knee"
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Internal knee angle (degrees)"
)

plt.title(
    f"{PARTICIPANT} first squat - "
    "corrected 3D knee angles"
)

plt.legend()

plt.tight_layout()

knee_plot = (
    OUT_DIR
    / (
        f"{PARTICIPANT}"
        "_corrected_knee_angles.png"
    )
)

plt.savefig(
    knee_plot,
    dpi=300
)

plt.close()


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    frames,
    left_hip_angle,
    label="Left hip"
)

plt.plot(
    frames,
    right_hip_angle,
    label="Right hip"
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Hip angle (degrees)"
)

plt.title(
    f"{PARTICIPANT} first squat - "
    "corrected hip angles"
)

plt.legend()

plt.tight_layout()

hip_plot = (
    OUT_DIR
    / (
        f"{PARTICIPANT}"
        "_corrected_hip_angles.png"
    )
)

plt.savefig(
    hip_plot,
    dpi=300
)

plt.close()


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    frames,
    root_to_foot_vertical_distance
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Vertical distance"
)

plt.title(
    f"{PARTICIPANT} first squat - "
    "root-to-feet vertical distance"
)

plt.tight_layout()

vertical_plot = (
    OUT_DIR
    / (
        f"{PARTICIPANT}"
        "_root_to_feet_vertical_distance.png"
    )
)

plt.savefig(
    vertical_plot,
    dpi=300
)

plt.close()


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    frames,
    foot_separation_horizontal,
    label="Foot separation"
)

plt.plot(
    frames,
    knee_separation_horizontal,
    label="Knee separation"
)

plt.plot(
    frames,
    hip_separation_horizontal,
    label="Hip separation"
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Horizontal X-Z distance"
)

plt.title(
    f"{PARTICIPANT} first squat - "
    "corrected horizontal joint separation"
)

plt.legend()

plt.tight_layout()

separation_plot = (
    OUT_DIR
    / (
        f"{PARTICIPANT}"
        "_corrected_joint_separation.png"
    )
)

plt.savefig(
    separation_plot,
    dpi=300
)

plt.close()


print("\n========================================")
print("SAVED")
print("========================================")

print(
    csv_path
)

print(
    knee_plot
)

print(
    hip_plot
)

print(
    vertical_plot
)

print(
    separation_plot
)

print(
    "\nDone."
)