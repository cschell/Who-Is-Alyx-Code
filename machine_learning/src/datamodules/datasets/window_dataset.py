from typing import Tuple

from src.datamodules.datasets.base_dataset import BaseDataset
from src.datamodules.datasets.helpers import compute_change_idxs
from src.datamodules.datasets.window_maker import WindowMaker


class WindowDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_frame_ids = False
        self.return_session_idx = False
        self.return_take_id = False

        self.window_size = self.data_hyperparameters.window_size
        self.frame_step_size = self.data_hyperparameters.frame_step_size

        self._load_and_set_data()
        self._load_window_maker()

        self._set_or_load_data_stats()

        self.frames = self._scale_data(self.frames)

    def _load_window_maker(self):
        change_idxs = compute_change_idxs(self.take_id)
        num_frames = len(self.take_id)
        self.wm = WindowMaker(num_frames, self.window_size, self.frame_step_size, change_idxs,
                              is_acceleration=self.data_hyperparameters.data_encoding.requires_acceleration_computation())

    @property
    def frame_ids(self):
        return self.wm.windowed_ids[:, 0]

    @property
    def num_features(self):
        return self.frames.shape[1]

    @property
    def num_samples(self):
        return self.wm.num_windows

    @property
    def sample_shape(self) -> Tuple[int, int]:
        return self.window_size, self.num_features

    @property
    def targets(self):
        return self.frame_targets[self.wm.first_window_frame_ids(slice(0, -1))]

    def __getitem__(self, window_idx):
        data = self.wm.to_windows(self.frames, window_idxs=[window_idx])[0]
        frame_id = self.wm.first_window_frame_ids(window_idx)
        target = self.frame_targets[frame_id]

        if len(data) == 0:
            raise IndexError

        item = {
            "data": data,
            "targets": target
        }

        if self.return_frame_ids:
            item["frame_id"] = frame_id.astype("int32")

        if self.return_session_idx:
            item["session_idx"] = self.session_idx[frame_id]

        if self.return_take_id:
            item["take_id"] = self.take_id[frame_id]

        return item
