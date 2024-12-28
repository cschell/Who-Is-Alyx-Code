# %%

import os

from rich.console import Console
from rich.progress import track
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import pathlib
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)

WHO_IS_ALYX_DATA_REPOSITORY_PATH = pathlib.Path("data/input/who-is-alyx/")
TARGET_FPS = 15
console = Console()

sessions_info = pd.read_csv(WHO_IS_ALYX_DATA_REPOSITORY_PATH.joinpath("summary.csv"))

relevant_subjects = sessions_info[sessions_info["total number of sessions"] == 2]

manual_start_timestamps  = {
    (44, "2022-05-25"): "2022-05-25 16:53:01.317911",
    (44, "2022-07-05"): "2022-07-05 11:26:50",
}


# %% Mappings

column_mappings = {
    'hmd_pos_x': "head_pos_x",
    'hmd_pos_y': "head_pos_y",
    'hmd_pos_z': "head_pos_z",
    'hmd_rot_x': "head_rot_x",
    'hmd_rot_y': "head_rot_y",
    'hmd_rot_z': "head_rot_z",
    'hmd_rot_w': "head_rot_w",

    'left_controller_pos_x': "left_hand_pos_x",
    'left_controller_pos_y': "left_hand_pos_y",
    'left_controller_pos_z': "left_hand_pos_z",
    'left_controller_rot_x': "left_hand_rot_x",
    'left_controller_rot_y': "left_hand_rot_y",
    'left_controller_rot_z': "left_hand_rot_z",
    'left_controller_rot_w': "left_hand_rot_w",

    'right_controller_pos_x': "right_hand_pos_x",
    'right_controller_pos_y': "right_hand_pos_y",
    'right_controller_pos_z': "right_hand_pos_z",
    'right_controller_rot_x': "right_hand_rot_x",
    'right_controller_rot_y': "right_hand_rot_y",
    'right_controller_rot_z': "right_hand_rot_z",
    'right_controller_rot_w': "right_hand_rot_w",
}

# %%

output_path = pathlib.Path(f"data/intermediate/{TARGET_FPS}_fps_data_{len(relevant_subjects)}_subjects_.hdf5")

if output_path.exists():
    os.remove(output_path)

select_features = list(column_mappings.values())
take_id = -1
session_id = -1

# %%

for idx, session_info in track(relevant_subjects.iterrows(), total=len(relevant_subjects)):
    player_dir = pathlib.Path(WHO_IS_ALYX_DATA_REPOSITORY_PATH.joinpath(f"players/{session_info['player_id']:02d}/"))

    for subject_session_idx, session_dir_path in enumerate(sorted(list(player_dir.glob("*")))):
        session_id += 1

        for take_idx, csv_path in enumerate(list(session_dir_path.glob("vr-controllers*.csv"))):
            take_id += 1

            if manual_start_timestamp := manual_start_timestamps.get((session_info["player_id"], session_dir_path.name)):
                start_timestamp = manual_start_timestamp
            else:
                eye_tracking = pd.read_csv(csv_path.with_name("eye-tracking.csv"), nrows=1, usecols=["timestamp"])
                start_timestamp = pd.to_datetime(eye_tracking["timestamp"][0])

            session_data = pd.read_csv(csv_path, index_col="delta_time_ms").rename(columns=column_mappings).sort_index()
            session_data["timestamp"] = pd.to_datetime(session_data["timestamp"])

            before_length = len(session_data)
            orig_session_start = session_data["timestamp"][0]
            session_data = session_data.query(f"timestamp >= '{start_timestamp}'")
            after_length = len(session_data)

            if before_length > after_length:
                length = session_data["timestamp"].iloc[-1] - session_data["timestamp"].iloc[0]
                # console.log(f"removed {before_length - after_length} frames from {csv_path}: start was {orig_session_start}, now is {start_timestamp}, duration is now {length}")
            session_data.index = pd.to_timedelta(session_data.index, unit="ms")

            features = session_data[select_features].copy()

            mspf = 1000 / TARGET_FPS

            original_index = features.index.total_seconds() * 1000
            target_index = np.arange(original_index.min(), original_index.max(), mspf)

            interpolated_features = pd.DataFrame(index=pd.TimedeltaIndex(target_index, name="timestamp", unit="ms"))

            positional_features = [c for c in features.columns if "_pos_" in c]

            features.loc[:, positional_features] = features[positional_features].interpolate("time")

            assert not any(features[positional_features].isna().any())

            for pos_feature_name in positional_features:
                column = features[pos_feature_name]
                interpolated_features[pos_feature_name] = np.interp(x=target_index, xp=original_index, fp=column)

            for joint in ["head", "left_hand", "right_hand"]:
                joint_orientation_features = [f"{joint}_rot_{c}" for c in "xyzw"]
                orientational_features = features[joint_orientation_features].dropna()
                rotations = R.from_quat(orientational_features)
                dropped_na_original_index = orientational_features.index.total_seconds() * 1000
                slerp = Slerp(dropped_na_original_index, rotations)
                interpolated_features[joint_orientation_features] = slerp(target_index).as_quat()

            interpolated_features["take_id"] = take_id
            interpolated_features["session_id"] = session_id
            interpolated_features["frame_idx"] = np.arange(len(interpolated_features))
            interpolated_features["subject_id"] = session_info['player_id']
            interpolated_features["subject_session_idx"] = subject_session_idx
            interpolated_features["subject_take_idx"] = take_idx
            assert not any(interpolated_features.isna().any())
            interpolated_features.to_hdf(output_path, key=f"subject_session_idx_{subject_session_idx}", mode="a", index=False, dropna=True, append=True, min_itemsize=11)

console.log(f"aggregated data in {output_path}")
