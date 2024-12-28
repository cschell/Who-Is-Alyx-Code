import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation


def process_rotation_to_6d_representation(frames: pd.DataFrame):
    rot_x_keys = [col[:-1] for col in frames.columns if '_rot_x' in col]
    transformed_df = frames.copy()
    for rot_key in rot_x_keys:
        keys = [rot_key + c for c in "xyzw"]
        non_na_frames = frames.loc[frames[keys].notna().all(axis=1), keys]
        if len(non_na_frames) > 0:
            rotations = Rotation.from_quat(non_na_frames)
            transformed_rotations = rotation_to_6d(rotations).reshape(-1, 6)
            transformed_df.drop(keys, axis=1, inplace=True)
            for i in range(6):
                new_col_key = f"{rot_key}6d_{i + 1}"
                transformed_df[new_col_key] = np.nan
                transformed_df.loc[non_na_frames.index, new_col_key] = transformed_rotations[:, i]
    return transformed_df


def rotation_to_6d(rot: Rotation) -> np.ndarray:
    matrix = rot.as_matrix()
    if len(matrix.shape) == 3:
        return matrix[:, :, :-1]
    return matrix[:, :-1]
