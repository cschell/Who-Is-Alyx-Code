import os
import pathlib
from typing import Optional
import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader

from src.datamodules.datasets.window_dataset import WindowDataset

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))


class RetrievalDatamodule(LightningDataModule):
    train_dataset: WindowDataset = None
    val_dataset: WindowDataset = None
    test_dataset: WindowDataset = None

    default_dataloader_settings = dict(
            num_workers=NUM_WORKERS,
            pin_memory=True,
    )

    def __init__(self, data_path: str, batch_size: int, seed: int, split: dict, data_stats_path, data_hyperparameters, data_loader_settings=None, dataset_kwargs=None):
        super().__init__()
        if dataset_kwargs:
            self.dataset_kwargs = dataset_kwargs
        else:
            self.dataset_kwargs = {}
        self.data_stats_path = pathlib.Path(data_stats_path)

        if not data_loader_settings:
            data_loader_settings = {}

        self.data_loader_settings = {**self.default_dataloader_settings, **data_loader_settings}

        self.batch_size = batch_size
        self.data_path = pathlib.Path(data_path)
        self.data_hyperparameters = data_hyperparameters
        self.seed = seed
        self.split = split
        self._subject_to_dataset_mapping()

    def _subject_to_dataset_mapping(self):
        np.random.seed(self.seed)
        subject_ids = pd.read_hdf(self.data_path, key="subject_ids").values.flatten()
        np.random.shuffle(subject_ids)
        train_no = self.split.get("train")
        val_no = self.split.get("validation")
        test_no = self.split.get("test")
        assert len(subject_ids) >= train_no + val_no + test_no, \
            f"The number of subjects does not match the sum of the subjects specified in the split dictionary. " \
            f"Split dictionary has to be a max of {len(subject_ids)} subjects."

        self.train_subject_ids = subject_ids[:train_no].tolist()

        if test_no:
            self.validation_subject_ids = subject_ids[-(val_no + test_no):-test_no].tolist()
        else:
            self.validation_subject_ids = subject_ids[-val_no:].tolist()

        self.test_subject_ids = subject_ids[-test_no:].tolist()

        # for verification:
        """
        output_path_data = pathlib.Path(f"/storage/taf/half_life_alyx/verification_new_split_subjects.hdf5")
        if output_path_data.exists():
            os.remove(output_path_data)
        pd.DataFrame({"train": self.train_subject_ids}).to_hdf(output_path_data, key="train", mode="a", index=False, dropna=True, append=True)
        pd.DataFrame({"validation": self.validation_subject_ids}).to_hdf(output_path_data, key="validation", mode="a", index=False, dropna=True, append=True)
        pd.DataFrame({"test": self.test_subject_ids}).to_hdf(output_path_data, key="test", mode="a", index=False, dropna=True, append=True)
        print("saved train, val, test subjects in /storage/taf/half_life_alyx/verification_new_split_subjects.hdf5")
        """

    @property
    def samples_per_class_per_batch(self):
        return self.batch_size // self.train_dataset.num_classes

    def setup(self, stage: Optional[str] = "train"):
        if stage == "test":
            self.test_dataset = WindowDataset(self.data_path,
                                              self.test_subject_ids,
                                              enforce_data_stats=self.data_stats_path,
                                              data_hyperparameters=self.data_hyperparameters,
                                              **self.dataset_kwargs)
            self.test_dataset.return_session_idx = True
            self.test_dataset.return_frame_ids = True
        elif stage == "validate":
            self.val_dataset = WindowDataset(self.data_path,
                                             self.validation_subject_ids,
                                             enforce_data_stats=self.data_stats_path,
                                             data_hyperparameters=self.data_hyperparameters,
                                             **self.dataset_kwargs)
        else:
            if not self.train_dataset:
                self.train_dataset = WindowDataset(self.data_path, self.train_subject_ids,
                                                   data_hyperparameters=self.data_hyperparameters,
                                                   **self.dataset_kwargs)
                self.train_dataset.safe_settings(self.data_stats_path)

            if not self.val_dataset:
                self.val_dataset = WindowDataset(self.data_path, self.validation_subject_ids,
                                                 enforce_data_stats=self.train_dataset.data_stats,
                                                 data_hyperparameters=self.data_hyperparameters,
                                                 **self.dataset_kwargs)
                self.val_dataset.return_session_idx = True
                self.val_dataset.return_take_id = True

    def train_dataloader(self):
        settings = self.data_loader_settings.copy()
        settings["batch_size"] = self.batch_size
        settings["sampler"] = MPerClassSampler(labels=self.train_dataset.targets,
                                               m=self.samples_per_class_per_batch,
                                               length_before_new_iter=1_000_000)
        return DataLoader(self.train_dataset, **settings)

    def val_dataloader(self):
        settings = self.data_loader_settings.copy()
        settings["batch_size"] = self.batch_size
        return DataLoader(self.val_dataset, **settings)

    def test_dataloader(self):
        settings = self.data_loader_settings.copy()
        settings["batch_size"] = self.batch_size
        return DataLoader(self.test_dataset, **settings)

    def safe_train_data_stats_settings(self):
        self.train_dataset.safe_settings(self.data_stats_path)
