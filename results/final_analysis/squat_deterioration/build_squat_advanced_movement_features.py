from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")

COHORT_PATH = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\squat_longitudinal_change_dataset.csv"
)

OUTPUT_DIR = Path(
    r"C:\Masters\results\final_analysis\squat_deterioration"
    r"\advanced_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEATURE_OUTPUT = (
    OUTPUT_DIR
    / "squat_advanced_movement_features.csv"
)

REP_OUTPUT = (
    OUTPUT_DIR
    / "squat_advanced_rep_features.csv"
)

QC_OUTPUT = (
    OUTPUT_DIR
    / "squat_advanced_feature_qc.csv"
)

FORMULA_OUTPUT = (
    OUTPUT_DIR
    / "advanced_feature_formula_register.csv"
)


FPS = 30.0
DT = 1.0 / FPS

SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3


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


def valid_savgol_window(length, requested):
    if length < 5:
        return None

    window = min(
        requested,
        length if length % 2 == 1 else length - 1
    )

    if window <= SAVGOL_POLYORDER:
        window = SAVGOL_POLYORDER + 2

        if window % 2 == 0:
            window += 1

    if window > length:
        window = length if length % 2 == 1 else length - 1

    if window <= SAVGOL_POLYORDER:
        return None

    return window


def smooth_array(values):
    values = np.asarray(
        values,
        dtype=float
    )

    window = valid_savgol_window(
        len(values),
        SAVGOL_WINDOW
    )

    if window is None:
        return values.copy()

    return savgol_filter(
        values,
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        axis=0,
        mode="interp"
    )


def vector_angle(a, b):
    numerator = np.sum(
        a * b,
        axis=-1
    )

    denominator = (
        np.linalg.norm(a, axis=-1)
        *
        np.linalg.norm(b, axis=-1)
    )

    cosine = np.divide(
        numerator,
        denominator,
        out=np.full(
            np.shape(numerator),
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


def angle_3d(a, b, c):
    return vector_angle(
        a - b,
        c - b
    )


def horizontal_distance(a, b):
    dx = (
        a[:, X_AXIS]
        -
        b[:, X_AXIS]
    )

    dz = (
        a[:, Z_AXIS]
        -
        b[:, Z_AXIS]
    )

    return np.sqrt(
        dx ** 2
        +
        dz ** 2
    )


def safe_ratio(a, b):
    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    return np.divide(
        a,
        b,
        out=np.full(
            np.broadcast(a, b).shape,
            np.nan,
            dtype=float
        ),
        where=np.abs(b) > 1e-12
    )


def rms(values):
    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(
                values ** 2
            )
        )
    )


def coefficient_of_variation(values):
    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan

    mean_value = np.mean(values)

    if abs(mean_value) < 1e-12:
        return np.nan

    return float(
        np.std(
            values,
            ddof=1
        )
        /
        abs(mean_value)
        *
        100.0
    )


def summary_stats(values, prefix):
    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_rms": np.nan,
        }

    return {
        f"{prefix}_mean":
            float(np.mean(values)),

        f"{prefix}_median":
            float(np.median(values)),

        f"{prefix}_std":
            float(
                np.std(
                    values,
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0,

        f"{prefix}_min":
            float(np.min(values)),

        f"{prefix}_max":
            float(np.max(values)),

        f"{prefix}_range":
            float(
                np.max(values)
                -
                np.min(values)
            ),

        f"{prefix}_iqr":
            float(
                np.percentile(
                    values,
                    75
                )
                -
                np.percentile(
                    values,
                    25
                )
            ),

        f"{prefix}_rms":
            rms(values),
    }


def derivative_trajectory(position):
    velocity = np.gradient(
        position,
        DT,
        axis=0
    )

    acceleration = np.gradient(
        velocity,
        DT,
        axis=0
    )

    jerk = np.gradient(
        acceleration,
        DT,
        axis=0
    )

    return (
        velocity,
        acceleration,
        jerk
    )


def magnitude(vector):
    return np.linalg.norm(
        vector,
        axis=-1
    )


def path_length(trajectory):
    trajectory = np.asarray(
        trajectory,
        dtype=float
    )

    if len(trajectory) < 2:
        return np.nan

    differences = np.diff(
        trajectory,
        axis=0
    )

    return float(
        np.sum(
            np.linalg.norm(
                differences,
                axis=1
            )
        )
    )


def cyclic_path_efficiency(trajectory):
    trajectory = np.asarray(
        trajectory,
        dtype=float
    )

    if len(trajectory) < 3:
        return np.nan

    actual_path = path_length(
        trajectory
    )

    if (
        not np.isfinite(actual_path)
        or actual_path <= 1e-12
    ):
        return np.nan

    start = trajectory[0]
    end = trajectory[-1]

    distance_from_start = np.linalg.norm(
        trajectory - start,
        axis=1
    )

    farthest_index = int(
        np.argmax(
            distance_from_start
        )
    )

    farthest = trajectory[
        farthest_index
    ]

    reference_path = (
        np.linalg.norm(
            farthest - start
        )
        +
        np.linalg.norm(
            end - farthest
        )
    )

    return float(
        reference_path
        /
        actual_path
    )


def log_dimensionless_jerk_scalar(signal):
    signal = np.asarray(
        signal,
        dtype=float
    )

    signal = signal[
        np.isfinite(signal)
    ]

    if len(signal) < 7:
        return np.nan

    duration = (
        len(signal) - 1
    ) * DT

    amplitude = (
        np.max(signal)
        -
        np.min(signal)
    )

    if (
        duration <= 0
        or amplitude <= 1e-12
    ):
        return np.nan

    velocity = np.gradient(
        signal,
        DT
    )

    acceleration = np.gradient(
        velocity,
        DT
    )

    jerk = np.gradient(
        acceleration,
        DT
    )

    jerk_integral = np.trapezoid(
        jerk ** 2,
        dx=DT
    )

    dimensionless_jerk = (
        duration ** 5
        /
        amplitude ** 2
        *
        jerk_integral
    )

    if (
        not np.isfinite(
            dimensionless_jerk
        )
        or dimensionless_jerk <= 0
    ):
        return np.nan

    return float(
        -np.log(
            dimensionless_jerk
        )
    )


def log_dimensionless_jerk_trajectory(trajectory):
    trajectory = np.asarray(
        trajectory,
        dtype=float
    )

    if len(trajectory) < 7:
        return np.nan

    duration = (
        len(trajectory) - 1
    ) * DT

    start = trajectory[0]

    amplitude = np.max(
        np.linalg.norm(
            trajectory - start,
            axis=1
        )
    )

    if (
        duration <= 0
        or amplitude <= 1e-12
    ):
        return np.nan

    velocity = np.gradient(
        trajectory,
        DT,
        axis=0
    )

    acceleration = np.gradient(
        velocity,
        DT,
        axis=0
    )

    jerk = np.gradient(
        acceleration,
        DT,
        axis=0
    )

    jerk_squared = np.sum(
        jerk ** 2,
        axis=1
    )

    jerk_integral = np.trapezoid(
        jerk_squared,
        dx=DT
    )

    dimensionless_jerk = (
        duration ** 5
        /
        amplitude ** 2
        *
        jerk_integral
    )

    if (
        not np.isfinite(
            dimensionless_jerk
        )
        or dimensionless_jerk <= 0
    ):
        return np.nan

    return float(
        -np.log(
            dimensionless_jerk
        )
    )


def estimate_period(signal):
    signal = np.asarray(
        signal,
        dtype=float
    )

    if len(signal) < 30:
        return np.nan

    centred = (
        signal
        -
        np.mean(signal)
    )

    autocorrelation = np.correlate(
        centred,
        centred,
        mode="full"
    )

    autocorrelation = (
        autocorrelation[
            len(signal) - 1:
        ]
    )

    minimum_lag = int(
        0.5 * FPS
    )

    maximum_lag = min(
        int(
            4.0 * FPS
        ),
        len(signal) - 1
    )

    if maximum_lag <= minimum_lag:
        return np.nan

    section = autocorrelation[
        minimum_lag:
        maximum_lag + 1
    ]

    period = (
        np.argmax(section)
        +
        minimum_lag
    )

    return int(period)


def find_rep_peaks(
    signal,
    expected_reps,
    estimated_period
):
    signal = np.asarray(
        signal,
        dtype=float
    )

    if (
        not np.isfinite(
            estimated_period
        )
        or estimated_period <= 0
    ):
        minimum_distance = int(
            FPS * 0.5
        )
    else:
        minimum_distance = max(
            5,
            int(
                estimated_period
                *
                0.55
            )
        )

    signal_std = np.std(
        signal
    )

    if signal_std <= 1e-12:
        return (
            np.array(
                [],
                dtype=int
            ),
            1
        )

    candidates = []

    for orientation in [
        1,
        -1
    ]:

        oriented = (
            signal
            *
            orientation
        )

        for prominence_factor in [
            0.30,
            0.20,
            0.10,
            0.05,
            0.0,
        ]:

            prominence = (
                signal_std
                *
                prominence_factor
            )

            peaks, properties = find_peaks(
                oriented,
                distance=minimum_distance,
                prominence=prominence
                if prominence > 0
                else None
            )

            original_count = len(
                peaks
            )

            if (
                expected_reps
                and original_count
                > expected_reps
            ):

                prominences = properties.get(
                    "prominences",
                    np.abs(
                        oriented[
                            peaks
                        ]
                    )
                )

                keep = np.argsort(
                    prominences
                )[
                    -expected_reps:
                ]

                peaks = np.sort(
                    peaks[
                        keep
                    ]
                )

            difference = abs(
                len(peaks)
                -
                expected_reps
            )

            if len(peaks) > 1:
                intervals = np.diff(
                    peaks
                )

                interval_variation = (
                    np.std(intervals)
                    /
                    np.mean(intervals)
                )
            else:
                interval_variation = np.inf

            candidates.append(
                (
                    difference,
                    interval_variation,
                    peaks,
                    orientation,
                )
            )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1]
        )
    )

    best = candidates[0]

    return (
        best[2],
        best[3]
    )


def repetition_signal(relative_pose):
    matrix = relative_pose[
        :,
        1:,
        :
    ].reshape(
        len(relative_pose),
        -1
    )

    valid_columns = (
        np.nanstd(
            matrix,
            axis=0
        )
        >
        1e-8
    )

    matrix = matrix[
        :,
        valid_columns
    ]

    if matrix.shape[1] < 2:
        return None

    scaler = StandardScaler()

    standardised = scaler.fit_transform(
        matrix
    )

    principal_component = PCA(
        n_components=1
    ).fit_transform(
        standardised
    )[:, 0]

    principal_component = smooth_array(
        principal_component
    )

    return principal_component


def rep_boundaries_from_peaks(
    peaks,
    signal_length
):
    peaks = np.asarray(
        peaks,
        dtype=int
    )

    if len(peaks) == 0:
        return []

    boundaries = [0]

    for first, second in zip(
        peaks[:-1],
        peaks[1:]
    ):
        boundaries.append(
            int(
                round(
                    (
                        first
                        +
                        second
                    )
                    /
                    2
                )
            )
        )

    boundaries.append(
        signal_length
    )

    repetitions = []

    for index in range(
        len(peaks)
    ):
        start = boundaries[
            index
        ]

        end = boundaries[
            index + 1
        ]

        if (
            end
            -
            start
            >= 5
        ):
            repetitions.append(
                (
                    start,
                    end,
                    int(
                        peaks[index]
                    )
                )
            )

    return repetitions


def add_rep_summary(
    attempt_features,
    rep_table,
    column,
    prefix
):
    if column not in rep_table.columns:
        return

    values = pd.to_numeric(
        rep_table[column],
        errors="coerce"
    ).dropna().to_numpy()

    if len(values) == 0:
        return

    attempt_features.update(
        summary_stats(
            values,
            prefix
        )
    )

    attempt_features[
        f"{prefix}_cv_pct"
    ] = coefficient_of_variation(
        values
    )

    if len(values) >= 2:
        attempt_features[
            f"{prefix}_first_to_last_change"
        ] = float(
            values[-1]
            -
            values[0]
        )

        if abs(
            values[0]
        ) > 1e-12:
            attempt_features[
                f"{prefix}_first_to_last_pct_change"
            ] = float(
                (
                    values[-1]
                    -
                    values[0]
                )
                /
                abs(
                    values[0]
                )
                *
                100.0
            )
        else:
            attempt_features[
                f"{prefix}_first_to_last_pct_change"
            ] = np.nan


cohort = pd.read_csv(
    COHORT_PATH
)


if "start_time" in cohort.columns:
    START_COLUMN = "start_time"
elif "start_frame" in cohort.columns:
    START_COLUMN = "start_frame"
else:
    raise ValueError(
        "No start frame column found."
    )


if "end_time" in cohort.columns:
    END_COLUMN = "end_time"
elif "end_frame" in cohort.columns:
    END_COLUMN = "end_frame"
else:
    raise ValueError(
        "No end frame column found."
    )


required_columns = [
    "participant",
    "squat_segment_number",
    START_COLUMN,
    END_COLUMN,
    "reps",
]


missing = [
    column
    for column in required_columns
    if column not in cohort.columns
]


if missing:
    raise ValueError(
        f"Missing cohort columns: {missing}"
    )


formula_rows = [
    {
        "feature_family":
            "Knee internal angle",

        "calculation":
            "acos(((hip-knee) dot (foot-knee)) / "
            "(||hip-knee|| * ||foot-knee||))",

        "units":
            "degrees",

        "notes":
            "Calculated separately for left and right knee."
    },
    {
        "feature_family":
            "Knee flexion",

        "calculation":
            "180 - knee internal angle",

        "units":
            "degrees",

        "notes":
            "Larger value represents greater knee flexion."
    },
    {
        "feature_family":
            "Hip angle proxy",

        "calculation":
            "acos(((thorax-hip) dot (knee-hip)) / "
            "(||thorax-hip|| * ||knee-hip||))",

        "units":
            "degrees",

        "notes":
            "Body-pose-derived hip flexion proxy."
    },
    {
        "feature_family":
            "Bilateral asymmetry",

        "calculation":
            "100 * |left-right| / ((|left|+|right|)/2)",

        "units":
            "percent",

        "notes":
            "Calculated where the denominator is non-zero."
    },
    {
        "feature_family":
            "Horizontal separation",

        "calculation":
            "sqrt((x_left-x_right)^2 + (z_left-z_right)^2)",

        "units":
            "pose coordinate units",

        "notes":
            "Y is treated as the vertical axis."
    },
    {
        "feature_family":
            "Knee-to-stance ratio",

        "calculation":
            "horizontal knee separation / horizontal foot separation",

        "units":
            "ratio",

        "notes":
            "Used as a knee-separation / valgus-type proxy only."
    },
    {
        "feature_family":
            "Relative squat depth proxy",

        "calculation":
            "|hip-root vertical coordinate - mean foot vertical coordinate|",

        "units":
            "pose coordinate units",

        "notes":
            "Body-relative posture measure, not laboratory absolute squat depth."
    },
    {
        "feature_family":
            "Velocity",

        "calculation":
            "dp/dt",

        "units":
            "pose units per second",

        "notes":
            "Calculated using numerical gradient at 30 Hz."
    },
    {
        "feature_family":
            "Acceleration",

        "calculation":
            "d2p/dt2",

        "units":
            "pose units per second squared",

        "notes":
            "Second numerical derivative of smoothed position."
    },
    {
        "feature_family":
            "Jerk",

        "calculation":
            "d3p/dt3",

        "units":
            "pose units per second cubed",

        "notes":
            "Third numerical derivative of smoothed position."
    },
    {
        "feature_family":
            "Path length",

        "calculation":
            "sum(||p[t]-p[t-1]||)",

        "units":
            "pose coordinate units",

        "notes":
            "Total trajectory distance."
    },
    {
        "feature_family":
            "Cyclic path efficiency",

        "calculation":
            "(distance start-to-farthest + distance farthest-to-end) "
            "/ actual path length",

        "units":
            "ratio",

        "notes":
            "Designed for repeated out-and-back motion."
    },
    {
        "feature_family":
            "Log dimensionless jerk",

        "calculation":
            "-ln((T^5/A^2) * integral(jerk^2 dt))",

        "units":
            "dimensionless",

        "notes":
            "A is movement amplitude. Larger values indicate smoother motion."
    },
    {
        "feature_family":
            "Repetition duration",

        "calculation":
            "(rep end frame - rep start frame) / 30",

        "units":
            "seconds",

        "notes":
            "Repetitions segmented from PCA periodic motion signal."
    },
    {
        "feature_family":
            "Rep consistency",

        "calculation":
            "100 * standard deviation / absolute mean",

        "units":
            "percent CV",

        "notes":
            "Applied to duration, depth, knee flexion and other rep metrics."
    },
]


pd.DataFrame(
    formula_rows
).to_csv(
    FORMULA_OUTPUT,
    index=False
)


attempt_rows = []
rep_rows = []
qc_rows = []


for _, row in cohort.iterrows():

    participant = str(
        row["participant"]
    )

    attempt_number = int(
        row[
            "squat_segment_number"
        ]
    )

    start_frame = int(
        round(
            float(
                row[
                    START_COLUMN
                ]
            )
        )
    )

    end_frame = int(
        round(
            float(
                row[
                    END_COLUMN
                ]
            )
        )
    )

    expected_reps = int(
        round(
            float(
                row["reps"]
            )
        )
    )


    pose_path = (
        RAW_DIR
        / participant
        / f"{participant}_pose_3d.npy"
    )


    if not pose_path.exists():
        print(
            "Missing pose:",
            pose_path
        )
        continue


    pose_raw = np.load(
        pose_path,
        allow_pickle=True
    )


    segment = pose_raw[
        :,
        start_frame:end_frame,
        1:
    ]


    if (
        segment.shape[0] != 3
        or segment.shape[2] != 17
    ):
        print(
            "Unexpected pose shape:",
            participant,
            attempt_number,
            segment.shape
        )
        continue


    xyz = np.transpose(
        segment,
        (1, 2, 0)
    ).astype(float)


    finite_percentage = (
        np.isfinite(xyz).sum()
        /
        xyz.size
        *
        100.0
    )


    xyz_smoothed = smooth_array(
        xyz
    )


    root = xyz_smoothed[
        :,
        JOINTS["hip_root"],
        :
    ]


    relative_pose = (
        xyz_smoothed
        -
        root[:, None, :]
    )


    right_hip = xyz_smoothed[
        :,
        JOINTS["right_hip"],
        :
    ]

    right_knee = xyz_smoothed[
        :,
        JOINTS["right_knee"],
        :
    ]

    right_foot = xyz_smoothed[
        :,
        JOINTS["right_foot"],
        :
    ]


    left_hip = xyz_smoothed[
        :,
        JOINTS["left_hip"],
        :
    ]

    left_knee = xyz_smoothed[
        :,
        JOINTS["left_knee"],
        :
    ]

    left_foot = xyz_smoothed[
        :,
        JOINTS["left_foot"],
        :
    ]


    thorax = xyz_smoothed[
        :,
        JOINTS["thorax"],
        :
    ]


    right_knee_internal = angle_3d(
        right_hip,
        right_knee,
        right_foot
    )

    left_knee_internal = angle_3d(
        left_hip,
        left_knee,
        left_foot
    )


    right_knee_flexion = (
        180.0
        -
        right_knee_internal
    )

    left_knee_flexion = (
        180.0
        -
        left_knee_internal
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


    knee_angle_difference = np.abs(
        left_knee_flexion
        -
        right_knee_flexion
    )


    knee_asymmetry_pct = (
        100.0
        *
        safe_ratio(
            knee_angle_difference,
            (
                np.abs(
                    left_knee_flexion
                )
                +
                np.abs(
                    right_knee_flexion
                )
            )
            /
            2.0
        )
    )


    hip_angle_difference = np.abs(
        left_hip_angle
        -
        right_hip_angle
    )


    hip_asymmetry_pct = (
        100.0
        *
        safe_ratio(
            hip_angle_difference,
            (
                np.abs(
                    left_hip_angle
                )
                +
                np.abs(
                    right_hip_angle
                )
            )
            /
            2.0
        )
    )


    foot_separation = horizontal_distance(
        left_foot,
        right_foot
    )

    knee_separation = horizontal_distance(
        left_knee,
        right_knee
    )

    hip_separation = horizontal_distance(
        left_hip,
        right_hip
    )


    knee_to_stance_ratio = safe_ratio(
        knee_separation,
        foot_separation
    )


    knee_to_hip_ratio = safe_ratio(
        knee_separation,
        hip_separation
    )


    mean_foot_vertical = (
        left_foot[:, Y_AXIS]
        +
        right_foot[:, Y_AXIS]
    ) / 2.0


    relative_depth_signal = np.abs(
        root[:, Y_AXIS]
        -
        mean_foot_vertical
    )


    trunk_vector = (
        thorax
        -
        root
    )


    vertical_reference = np.zeros_like(
        trunk_vector
    )

    vertical_reference[
        :,
        Y_AXIS
    ] = 1.0


    trunk_angle = vector_angle(
        trunk_vector,
        vertical_reference
    )


    frame_displacement = np.linalg.norm(
        np.diff(
            xyz_smoothed,
            axis=0
        ),
        axis=2
    )


    frame_mean_displacement = np.mean(
        frame_displacement,
        axis=1
    )


    attempt_features = {
        "participant":
            participant,

        "squat_segment_number":
            attempt_number,

        "start_frame":
            start_frame,

        "end_frame":
            end_frame,

        "duration_frames":
            end_frame
            -
            start_frame,

        "duration_seconds":
            (
                end_frame
                -
                start_frame
            )
            /
            FPS,

        "labelled_reps":
            expected_reps,

        "pose_finite_pct":
            finite_percentage,

        "pose_frame_jump_mean":
            float(
                np.mean(
                    frame_mean_displacement
                )
            ),

        "pose_frame_jump_p95":
            float(
                np.percentile(
                    frame_mean_displacement,
                    95
                )
            ),

        "pose_frame_jump_max":
            float(
                np.max(
                    frame_mean_displacement
                )
            ),
    }


    attempt_features.update(
        summary_stats(
            left_knee_internal,
            "left_knee_internal_angle"
        )
    )

    attempt_features.update(
        summary_stats(
            right_knee_internal,
            "right_knee_internal_angle"
        )
    )

    attempt_features.update(
        summary_stats(
            left_knee_flexion,
            "left_knee_flexion"
        )
    )

    attempt_features.update(
        summary_stats(
            right_knee_flexion,
            "right_knee_flexion"
        )
    )

    attempt_features.update(
        summary_stats(
            left_hip_angle,
            "left_hip_angle_proxy"
        )
    )

    attempt_features.update(
        summary_stats(
            right_hip_angle,
            "right_hip_angle_proxy"
        )
    )

    attempt_features.update(
        summary_stats(
            knee_angle_difference,
            "knee_flexion_abs_difference"
        )
    )

    attempt_features.update(
        summary_stats(
            knee_asymmetry_pct,
            "knee_flexion_asymmetry_pct"
        )
    )

    attempt_features.update(
        summary_stats(
            hip_angle_difference,
            "hip_angle_abs_difference"
        )
    )

    attempt_features.update(
        summary_stats(
            hip_asymmetry_pct,
            "hip_angle_asymmetry_pct"
        )
    )

    attempt_features.update(
        summary_stats(
            foot_separation,
            "stance_width"
        )
    )

    attempt_features.update(
        summary_stats(
            knee_separation,
            "knee_separation"
        )
    )

    attempt_features.update(
        summary_stats(
            hip_separation,
            "hip_separation"
        )
    )

    attempt_features.update(
        summary_stats(
            knee_to_stance_ratio,
            "knee_to_stance_ratio"
        )
    )

    attempt_features.update(
        summary_stats(
            knee_to_hip_ratio,
            "knee_to_hip_ratio"
        )
    )

    attempt_features.update(
        summary_stats(
            relative_depth_signal,
            "relative_squat_depth"
        )
    )

    attempt_features.update(
        summary_stats(
            trunk_angle,
            "trunk_vertical_angle"
        )
    )


    attempt_features[
        "left_knee_flexion_ldlj"
    ] = log_dimensionless_jerk_scalar(
        left_knee_flexion
    )

    attempt_features[
        "right_knee_flexion_ldlj"
    ] = log_dimensionless_jerk_scalar(
        right_knee_flexion
    )

    attempt_features[
        "relative_squat_depth_ldlj"
    ] = log_dimensionless_jerk_scalar(
        relative_depth_signal
    )

    attempt_features[
        "trunk_angle_ldlj"
    ] = log_dimensionless_jerk_scalar(
        trunk_angle
    )


    all_joint_speed = []
    all_joint_acceleration = []
    all_joint_jerk = []


    for joint_name, joint_index in JOINTS.items():

        if joint_name == "hip_root":
            trajectory = xyz_smoothed[
                :,
                joint_index,
                :
            ]

            feature_prefix = (
                "global_hip_root"
            )

        else:
            trajectory = relative_pose[
                :,
                joint_index,
                :
            ]

            feature_prefix = joint_name


        velocity, acceleration, jerk = (
            derivative_trajectory(
                trajectory
            )
        )


        speed_magnitude = magnitude(
            velocity
        )

        acceleration_magnitude = magnitude(
            acceleration
        )

        jerk_magnitude = magnitude(
            jerk
        )


        attempt_features.update(
            summary_stats(
                speed_magnitude,
                f"{feature_prefix}_speed"
            )
        )

        attempt_features.update(
            summary_stats(
                acceleration_magnitude,
                f"{feature_prefix}_acceleration"
            )
        )

        attempt_features.update(
            summary_stats(
                jerk_magnitude,
                f"{feature_prefix}_jerk"
            )
        )


        attempt_features[
            f"{feature_prefix}_path_length"
        ] = path_length(
            trajectory
        )


        attempt_features[
            f"{feature_prefix}_ldlj"
        ] = (
            log_dimensionless_jerk_trajectory(
                trajectory
            )
        )


        if joint_name != "hip_root":
            all_joint_speed.append(
                speed_magnitude
            )

            all_joint_acceleration.append(
                acceleration_magnitude
            )

            all_joint_jerk.append(
                jerk_magnitude
            )


    all_joint_speed = np.vstack(
        all_joint_speed
    ).T

    all_joint_acceleration = np.vstack(
        all_joint_acceleration
    ).T

    all_joint_jerk = np.vstack(
        all_joint_jerk
    ).T


    attempt_features.update(
        summary_stats(
            np.mean(
                all_joint_speed,
                axis=1
            ),
            "whole_body_speed"
        )
    )

    attempt_features.update(
        summary_stats(
            np.mean(
                all_joint_acceleration,
                axis=1
            ),
            "whole_body_acceleration"
        )
    )

    attempt_features.update(
        summary_stats(
            np.mean(
                all_joint_jerk,
                axis=1
            ),
            "whole_body_jerk"
        )
    )


    rep_signal = repetition_signal(
        relative_pose
    )


    if rep_signal is None:
        estimated_period = np.nan

        peaks = np.array(
            [],
            dtype=int
        )

        peak_orientation = np.nan

    else:
        estimated_period = estimate_period(
            rep_signal
        )

        peaks, peak_orientation = (
            find_rep_peaks(
                rep_signal,
                expected_reps,
                estimated_period
            )
        )


    repetitions = (
        rep_boundaries_from_peaks(
            peaks,
            len(xyz_smoothed)
        )
    )


    detected_reps = len(
        repetitions
    )


    if detected_reps == expected_reps:
        rep_qc = "exact"

    elif abs(
        detected_reps
        -
        expected_reps
    ) == 1:
        rep_qc = "close"

    else:
        rep_qc = "review"


    attempt_features[
        "detected_reps"
    ] = detected_reps

    attempt_features[
        "rep_count_difference"
    ] = (
        detected_reps
        -
        expected_reps
    )

    attempt_features[
        "rep_period_estimate_frames"
    ] = estimated_period

    attempt_features[
        "rep_period_estimate_seconds"
    ] = (
        estimated_period
        /
        FPS
        if np.isfinite(
            estimated_period
        )
        else np.nan
    )


    current_rep_rows = []


    for rep_number, (
        rep_start,
        rep_end,
        rep_peak
    ) in enumerate(
        repetitions,
        start=1
    ):

        rep_slice = slice(
            rep_start,
            rep_end
        )


        rep_duration = (
            rep_end
            -
            rep_start
        ) / FPS


        rep_left_knee = (
            left_knee_flexion[
                rep_slice
            ]
        )

        rep_right_knee = (
            right_knee_flexion[
                rep_slice
            ]
        )

        rep_depth = (
            relative_depth_signal[
                rep_slice
            ]
        )

        rep_trunk = trunk_angle[
            rep_slice
        ]

        rep_stance = foot_separation[
            rep_slice
        ]

        rep_knee_separation = (
            knee_separation[
                rep_slice
            ]
        )

        rep_knee_ratio = (
            knee_to_stance_ratio[
                rep_slice
            ]
        )


        rep_row = {
            "participant":
                participant,

            "squat_segment_number":
                attempt_number,

            "rep_number":
                rep_number,

            "rep_start_frame_relative":
                rep_start,

            "rep_end_frame_relative":
                rep_end,

            "rep_peak_frame_relative":
                rep_peak,

            "rep_duration_seconds":
                rep_duration,

            "relative_depth_amplitude":
                float(
                    np.max(
                        rep_depth
                    )
                    -
                    np.min(
                        rep_depth
                    )
                ),

            "left_knee_max_flexion":
                float(
                    np.max(
                        rep_left_knee
                    )
                ),

            "right_knee_max_flexion":
                float(
                    np.max(
                        rep_right_knee
                    )
                ),

            "left_knee_flexion_range":
                float(
                    np.max(
                        rep_left_knee
                    )
                    -
                    np.min(
                        rep_left_knee
                    )
                ),

            "right_knee_flexion_range":
                float(
                    np.max(
                        rep_right_knee
                    )
                    -
                    np.min(
                        rep_right_knee
                    )
                ),

            "knee_max_flexion_asymmetry":
                float(
                    abs(
                        np.max(
                            rep_left_knee
                        )
                        -
                        np.max(
                            rep_right_knee
                        )
                    )
                ),

            "trunk_angle_range":
                float(
                    np.max(
                        rep_trunk
                    )
                    -
                    np.min(
                        rep_trunk
                    )
                ),

            "stance_width_mean":
                float(
                    np.mean(
                        rep_stance
                    )
                ),

            "stance_width_range":
                float(
                    np.max(
                        rep_stance
                    )
                    -
                    np.min(
                        rep_stance
                    )
                ),

            "knee_separation_mean":
                float(
                    np.mean(
                        rep_knee_separation
                    )
                ),

            "knee_to_stance_ratio_mean":
                float(
                    np.nanmean(
                        rep_knee_ratio
                    )
                ),

            "knee_to_stance_ratio_min":
                float(
                    np.nanmin(
                        rep_knee_ratio
                    )
                ),

            "relative_depth_ldlj":
                log_dimensionless_jerk_scalar(
                    rep_depth
                ),

            "left_knee_flexion_ldlj":
                log_dimensionless_jerk_scalar(
                    rep_left_knee
                ),

            "right_knee_flexion_ldlj":
                log_dimensionless_jerk_scalar(
                    rep_right_knee
                ),

            "trunk_angle_ldlj":
                log_dimensionless_jerk_scalar(
                    rep_trunk
                ),
        }


        for joint_name in [
            "thorax",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_foot",
            "right_foot",
        ]:

            joint_index = JOINTS[
                joint_name
            ]

            trajectory = relative_pose[
                rep_slice,
                joint_index,
                :
            ]

            rep_row[
                f"{joint_name}_path_length"
            ] = path_length(
                trajectory
            )

            rep_row[
                f"{joint_name}_path_efficiency"
            ] = cyclic_path_efficiency(
                trajectory
            )

            rep_row[
                f"{joint_name}_ldlj"
            ] = (
                log_dimensionless_jerk_trajectory(
                    trajectory
                )
            )


        current_rep_rows.append(
            rep_row
        )

        rep_rows.append(
            rep_row
        )


    rep_table = pd.DataFrame(
        current_rep_rows
    )


    if not rep_table.empty:

        add_rep_summary(
            attempt_features,
            rep_table,
            "rep_duration_seconds",
            "rep_duration"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "relative_depth_amplitude",
            "rep_depth"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "left_knee_max_flexion",
            "rep_left_knee_max_flexion"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "right_knee_max_flexion",
            "rep_right_knee_max_flexion"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "knee_max_flexion_asymmetry",
            "rep_knee_max_flexion_asymmetry"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "trunk_angle_range",
            "rep_trunk_angle_range"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "stance_width_mean",
            "rep_stance_width"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "knee_to_stance_ratio_min",
            "rep_knee_to_stance_ratio_min"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "relative_depth_ldlj",
            "rep_depth_ldlj"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "left_knee_flexion_ldlj",
            "rep_left_knee_ldlj"
        )

        add_rep_summary(
            attempt_features,
            rep_table,
            "right_knee_flexion_ldlj",
            "rep_right_knee_ldlj"
        )


        for joint_name in [
            "thorax",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_foot",
            "right_foot",
        ]:

            add_rep_summary(
                attempt_features,
                rep_table,
                f"{joint_name}_path_efficiency",
                f"rep_{joint_name}_path_efficiency"
            )

            add_rep_summary(
                attempt_features,
                rep_table,
                f"{joint_name}_ldlj",
                f"rep_{joint_name}_ldlj"
            )


    qc_rows.append({
        "participant":
            participant,

        "squat_segment_number":
            attempt_number,

        "expected_reps":
            expected_reps,

        "detected_reps":
            detected_reps,

        "rep_count_difference":
            detected_reps
            -
            expected_reps,

        "rep_detection_qc":
            rep_qc,

        "estimated_period_frames":
            estimated_period,

        "estimated_period_seconds":
            estimated_period
            /
            FPS
            if np.isfinite(
                estimated_period
            )
            else np.nan,

        "peak_orientation":
            peak_orientation,

        "finite_pose_pct":
            finite_percentage,

        "frame_jump_p95":
            float(
                np.percentile(
                    frame_mean_displacement,
                    95
                )
            ),

        "frame_jump_max":
            float(
                np.max(
                    frame_mean_displacement
                )
            ),
    })


    attempt_rows.append(
        attempt_features
    )


    print(
        f"{participant} "
        f"attempt {attempt_number}: "
        f"expected {expected_reps}, "
        f"detected {detected_reps}, "
        f"QC={rep_qc}"
    )


attempt_features_df = pd.DataFrame(
    attempt_rows
)

rep_features_df = pd.DataFrame(
    rep_rows
)

qc_df = pd.DataFrame(
    qc_rows
)


attempt_features_df.to_csv(
    FEATURE_OUTPUT,
    index=False
)

rep_features_df.to_csv(
    REP_OUTPUT,
    index=False
)

qc_df.to_csv(
    QC_OUTPUT,
    index=False
)


print("\n========================================")
print("ADVANCED FEATURE EXTRACTION COMPLETE")
print("========================================")

print(
    "Attempts:",
    len(
        attempt_features_df
    )
)

print(
    "Participants:",
    attempt_features_df[
        "participant"
    ].nunique()
)

print(
    "Advanced columns:",
    len(
        attempt_features_df.columns
    )
)

print(
    "Individual repetitions:",
    len(
        rep_features_df
    )
)


print("\n========================================")
print("REPETITION DETECTION QC")
print("========================================")

if not qc_df.empty:

    print(
        qc_df[
            "rep_detection_qc"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\nDetected - expected:"
    )

    print(
        qc_df[
            "rep_count_difference"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


print("\n========================================")
print("EXAMPLE FEATURES")
print("========================================")

preview_columns = [
    "participant",
    "squat_segment_number",

    "left_knee_flexion_mean",
    "right_knee_flexion_mean",

    "left_knee_flexion_range",
    "right_knee_flexion_range",

    "knee_flexion_asymmetry_pct_mean",

    "relative_squat_depth_range",

    "stance_width_mean",

    "knee_to_stance_ratio_mean",

    "trunk_vertical_angle_mean",

    "whole_body_acceleration_mean",
    "whole_body_jerk_mean",

    "relative_squat_depth_ldlj",

    "detected_reps",

    "rep_duration_cv_pct",
    "rep_depth_cv_pct",
]


available_preview = [
    column
    for column in preview_columns
    if column
    in attempt_features_df.columns
]


print(
    attempt_features_df[
        available_preview
    ]
    .head(15)
    .round(4)
    .to_string(
        index=False
    )
)


print("\nSaved:")
print(
    FEATURE_OUTPUT
)

print(
    REP_OUTPUT
)

print(
    QC_OUTPUT
)

print(
    FORMULA_OUTPUT
)

print("\nDone.")