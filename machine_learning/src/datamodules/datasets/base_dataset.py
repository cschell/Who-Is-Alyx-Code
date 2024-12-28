import json
from collections import Mapping
from typing import Dict, Union, Any

import numpy as np
import pandas as pd
import torch

from src.datamodules.datasets.helpers import compute_change_idxs, limit_data_per_subject, compute_relative_positions_and_rotations, compute_velocities_for_position_and_rotations
from src.hyperparameters.data_hyperparameters import BinDataHyperparameters, WindowDataHyperparameters
from src.utils import utils
from src.utils.rotation_helper import process_rotation_to_6d_representation

log = utils.get_logger(__name__)


class BaseDataset(torch.utils.data.Dataset):
    feature_columns = ['head_pos_x',
                       'head_pos_y',
                       'head_pos_z',
                       'head_rot_x',
                       'head_rot_y',
                       'head_rot_z',
                       'head_rot_w',
                       'left_hand_pos_x',
                       'left_hand_pos_y',
                       'left_hand_pos_z',
                       'left_hand_rot_x',
                       'left_hand_rot_y',
                       'left_hand_rot_z',
                       'left_hand_rot_w',
                       'right_hand_pos_x',
                       'right_hand_pos_y',
                       'right_hand_pos_z',
                       'right_hand_rot_x',
                       'right_hand_rot_y',
                       'right_hand_rot_z',
                       'right_hand_rot_w'
                       ]

    frame_step_size: int = "unset"

    def __init__(self, hdf5_file_path, dataset_keys: list,
                 data_hyperparameters: Union[WindowDataHyperparameters, BinDataHyperparameters],
                 enforce_data_stats: Dict[str, Any] = None,
                 coordinate_system=None,
                 limit_train_time=None,
                 train_time_seed=None,
                 debug_mode=False,
                 **kwargs):
        self.debug_mode = debug_mode
        self.hdf5_file_path = hdf5_file_path
        self.dataset_keys = dataset_keys
        self.data_hyperparameters = data_hyperparameters
        self.enforce_data_stats = enforce_data_stats
        self.coordinate_system = coordinate_system
        self.limit_train_time = limit_train_time
        self.train_time_seed = train_time_seed
        # TODO:
        super().__init__()

    def _set_or_load_data_stats(self):
        if self.enforce_data_stats:
            if isinstance(self.enforce_data_stats, Mapping):
                stats = self.enforce_data_stats
            else:
                stats = self.load_settings(self.enforce_data_stats)

            self.means = stats["means"]
            self.stds = stats["stds"]
            # self._labels = stats["labels"]
        else:
            self.means = self._own_means
            self.stds = self._own_stds

    def safe_settings(self, path):
        with open(path, "w") as json_file:
            json.dump({
                "means": self.means.tolist(),
                "stds": self.stds.tolist(),
                "labels": self.labels.tolist()
            }, json_file)

    def load_settings(self, path):
        settings = json.load(open(path, "r"))

        return {"means": np.array(settings["means"]),
                "stds": np.array(settings["stds"]),
                "labels": np.array(settings["labels"]), }

    @property
    def labels(self) -> np.ndarray:
        if not hasattr(self, "_labels"):
            self._labels = np.sort(np.unique(self.subject_id))

        return self._labels

    @property
    def label_idx_mapping(self) -> Dict[str, int]:
        if not hasattr(self, "_label_to_idx"):
            self._label_idx_mapping = {label: idx for idx, label in enumerate(self.labels)}

        return self._label_idx_mapping

    def encode_labels_to_idx(self, labels) -> np.ndarray:
        return np.array([self.label_idx_mapping[label] for label in labels], dtype="int16")

    def encode_labels(self, labels) -> np.ndarray:
        return self.labels_encoded[labels]

    @property
    def labels_encoded(self):
        if not hasattr(self, "_labels_encoded"):
            self._labels_encoded = np.eye(len(self.labels))

        return self._labels_encoded

    @property
    def loss_weights(self):
        if not hasattr(self, "_loss_weights"):
            target_counts = np.bincount(self.frame_targets)
            loss_weights = (target_counts / target_counts.max()) ** -1
            self._loss_weights = loss_weights / loss_weights.max()
        return self._loss_weights

    def __len__(self):
        return self.num_samples

    def _scale_data(self, X):
        return (X - self.means) / self.stds

    @property
    def num_classes(self):
        return len(self.labels)

    def _prepare_data(self, data):
        return torch.from_numpy(self._scale_data(data)).float()

    def _load_and_set_data(self):
        assert len(self.feature_columns) % 3 == 0
        assert isinstance(self.dataset_keys, list), "dataset_keys have to be of type 'list'."
        data_df: pd.DataFrame = pd.concat([pd.read_hdf(self.hdf5_file_path, key=str(subj_id)) for subj_id in self.dataset_keys])

        # limiting train data:
        if (self.limit_train_time is not None) and (self.dataset_keys[0] == "train"):
            log.warning(f"Training time is limited to {self.limit_train_time}")

            # subject 10 only has 2 minutes of train time!
            minimum_train_time = 27
            max_offset_value = minimum_train_time - self.limit_train_time

            # calculating random offset and train time in frames
            np.random.seed(self.train_time_seed)
            if max_offset_value > 0:
                train_time_offset_mins = np.random.randint(max_offset_value, size=1)[0]
            else:
                train_time_offset_mins = 0
            train_time_offset_frames = train_time_offset_mins * self.data_hyperparameters.fps * 60
            train_time_frames = self.limit_train_time * self.data_hyperparameters.fps * 60

            # cutting train data accordingly for every subject
            cut_data_list = []
            for subject_id, data in data_df.groupby("subject_id"):
                data_min = (len(data) / 60) / self.data_hyperparameters.fps
                assert len(data) >= minimum_train_time * 60 * 15, f"subject {subject_id} does only have {data_min} minutes of train data"
                cut_data = data.iloc[train_time_offset_frames:(train_time_offset_frames + train_time_frames)].reset_index(drop=True)
                assert len(cut_data) <= train_time_frames, "something went wrong when limiting the train data"
                cut_data_list.append(cut_data)
            data_df = pd.concat(cut_data_list, ignore_index=True)

        data_df = data_df.reset_index()

        if self.data_hyperparameters.displace_positional_data:
            # displace subjects by 1 on x and y axis
            log.info("displacing data!")
            xy_position_columns = [c for c in self.feature_columns if "_pos_x" in c or "_pos_y" in c]
            data_df.loc[:, xy_position_columns] += 10  # unit is ~10 cm, so this offsets each subject by about 1m

        frames = data_df[self.feature_columns]

        if self.debug_mode:
            self.subject_id = data_df.subject_id
            frames[:] = np.repeat(self.encode_labels_to_idx(data_df.subject_id)[:, None], frames.shape[1], axis=1)
        else:
            if self.data_hyperparameters.data_encoding.requires_relative_computation():
                log.info(f"computing relative positional data")
                frames = compute_relative_positions_and_rotations(frames,
                                                                  reference_joint="head",
                                                                  target_joints=["left_hand",
                                                                                 "right_hand"],
                                                                  coordinate_system=self.coordinate_system)

                if self.data_hyperparameters.data_encoding.requires_velocity_computation():
                    log.info(f"computing velocity data")
                    change_idxs = compute_change_idxs(data_df.take_id)
                    frames = compute_velocities_for_position_and_rotations(frames, frame_step_size=self.frame_step_size, change_idxs=change_idxs)

                    if self.data_hyperparameters.data_encoding.requires_acceleration_computation():
                        log.info(f"computing acceleration data")
                        frames = compute_velocities_for_position_and_rotations(frames, frame_step_size=self.frame_step_size, change_idxs=change_idxs)

                        if self.data_hyperparameters.data_encoding.requires_jerk_computation():
                            log.info(f"computing jerk data")
                            frames = compute_velocities_for_position_and_rotations(frames, frame_step_size=self.frame_step_size, change_idxs=change_idxs)

        if self.data_hyperparameters.use_6d_rotation:
            frames = process_rotation_to_6d_representation(frames)
            log.info(f"converted rotations to 6d representation. New shape of frames <{frames.shape}>")

        self._own_means = np.nanmean(frames, axis=0)
        self._own_stds = np.nanstd(frames, axis=0)

        self.subject_id = data_df.subject_id
        self.frame_targets = self.encode_labels_to_idx(self.subject_id)
        if "session_idx" in data_df:
            self.session_idx = data_df.session_idx
        if "device" in data_df:
            self.device = data_df.device
        self.frame_idx = data_df.frame_idx
        self.take_id = data_df.take_id
        self.frames = frames
        assert len(frames.index) == len(np.unique(frames.index))

    @property
    def data_stats(self):
        return {"means": self.means, "stds": self.stds, "labels": self.labels}
