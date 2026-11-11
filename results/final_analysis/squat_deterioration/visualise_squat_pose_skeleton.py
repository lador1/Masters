from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PARTICIPANT = "w01"

RAW_DIR = Path(
    r"C:\Masters\data\raw\mmfit"
)

OUT_DIR = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\advanced_feature_checks"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


SKELETON_CONNECTIONS = [
    (
        "hip_root",
        "right_hip"
    ),
    (
        "right_hip",
        "right_knee"
    ),
    (
        "right_knee",
        "right_foot"
    ),

    (
        "hip_root",
        "left_hip"
    ),
    (
        "left_hip",
        "left_knee"
    ),
    (
        "left_knee",
        "left_foot"
    ),

    (
        "hip_root",
        "spine"
    ),
    (
        "spine",
        "thorax"
    ),
    (
        "thorax",
        "neck_nose"
    ),
    (
        "neck_nose",
        "head"
    ),

    (
        "thorax",
        "left_shoulder"
    ),
    (
        "left_shoulder",
        "left_elbow"
    ),
    (
        "left_elbow",
        "left_wrist"
    ),

    (
        "thorax",
        "right_shoulder"
    ),
    (
        "right_shoulder",
        "right_elbow"
    ),
    (
        "right_elbow",
        "right_wrist"
    ),
]


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


squats = labels[
    labels["exercise"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("squats")
].copy()


if squats.empty:
    raise ValueError(
        f"No squats found for {PARTICIPANT}"
    )


squat = squats.iloc[0]

start = int(
    squat["start_frame"]
)

end = int(
    squat["end_frame"]
)


segment = pose_raw[
    :,
    start:end,
    1:
]


xyz = np.transpose(
    segment,
    (1, 2, 0)
)


if xyz.shape[1] != 17:
    raise ValueError(
        f"Expected 17 joints, found {xyz.shape[1]}"
    )


print(
    "Participant:",
    PARTICIPANT
)

print(
    "Squat start:",
    start
)

print(
    "Squat end:",
    end
)

print(
    "Segment frames:",
    len(xyz)
)

print(
    "Labelled reps:",
    squat["reps"]
)

print(
    "Segment shape:",
    xyz.shape
)


frames_to_plot = np.linspace(
    0,
    len(xyz) - 1,
    12,
    dtype=int
)


all_x = xyz[:, :, 0]
all_z = xyz[:, :, 1]
all_y = xyz[:, :, 2]


x_min = float(
    np.nanmin(all_x)
)

x_max = float(
    np.nanmax(all_x)
)

z_min = float(
    np.nanmin(all_z)
)

z_max = float(
    np.nanmax(all_z)
)

y_min = float(
    np.nanmin(all_y)
)

y_max = float(
    np.nanmax(all_y)
)


x_mid = (
    x_min + x_max
) / 2.0

z_mid = (
    z_min + z_max
) / 2.0

y_mid = (
    y_min + y_max
) / 2.0


largest_range = max(
    x_max - x_min,
    z_max - z_min,
    y_max - y_min
)


half_range = (
    largest_range
    / 2.0
    * 1.10
)


for frame_number in frames_to_plot:

    frame = xyz[
        frame_number
    ]

    fig = plt.figure(
        figsize=(9, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    for (
        joint_a,
        joint_b
    ) in SKELETON_CONNECTIONS:

        index_a = JOINTS[
            joint_a
        ]

        index_b = JOINTS[
            joint_b
        ]

        point_a = frame[
            index_a
        ]

        point_b = frame[
            index_b
        ]

        ax.plot(
            [
                point_a[0],
                point_b[0]
            ],
            [
                point_a[1],
                point_b[1]
            ],
            [
                point_a[2],
                point_b[2]
            ],
            linewidth=2
        )


    for name, index in JOINTS.items():

        point = frame[
            index
        ]

        ax.scatter(
            point[0],
            point[1],
            point[2],
            s=35
        )


        if name in [
            "hip_root",

            "right_hip",
            "right_knee",
            "right_foot",

            "left_hip",
            "left_knee",
            "left_foot",

            "thorax",

            "left_shoulder",
            "right_shoulder",
        ]:

            ax.text(
                point[0],
                point[1],
                point[2],
                name,
                fontsize=7
            )


    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Z"
    )

    ax.set_zlabel(
        "Y (vertical)"
    )


    ax.set_xlim(
        x_mid - half_range,
        x_mid + half_range
    )

    ax.set_ylim(
        z_mid - half_range,
        z_mid + half_range
    )

    ax.set_zlim(
        y_mid - half_range,
        y_mid + half_range
    )


    ax.set_box_aspect(
        (1, 1, 1)
    )


    ax.set_title(
        f"{PARTICIPANT} first squat\n"
        f"segment frame {frame_number} "
        f"(absolute frame {start + frame_number})"
    )


    ax.view_init(
        elev=15,
        azim=-70
    )


    plt.tight_layout()


    output_path = (
        OUT_DIR
        / (
            f"{PARTICIPANT}"
            f"_corrected_skeleton_frame_"
            f"{frame_number:04d}.png"
        )
    )


    plt.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Saved:",
        output_path
    )


print(
    "\nDone."
)