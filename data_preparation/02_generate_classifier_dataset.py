import os
import pathlib
import numpy as np
import pandas as pd


OUTPUT_DIR = pathlib.Path("./data/output")
FPS = 15
NUMBER_OF_SUBJECTS = 71
validation_interval_in_minutes = 5 # in minutes

session_1_data = pd.read_hdf(OUTPUT_DIR.joinpath(f"{FPS}_fps_data_{NUMBER_OF_SUBJECTS}_subjects.hdf5"),
                             key="subject_session_idx_0")
session_2_data = pd.read_hdf(OUTPUT_DIR.joinpath(f"{FPS}_fps_data_{NUMBER_OF_SUBJECTS}_subjects.hdf5"),
                             key="subject_session_idx_1")



OFFSET_FRAMES = FPS * 60 * 1
CUTOFF_FRAMES = FPS * 60 * 1

num_of_frames_in_validation_data = FPS * 60 * validation_interval_in_minutes

np.random.seed(42)

subject_ids = session_1_data.subject_id.unique()
np.random.shuffle(subject_ids)

assert len(subject_ids) == NUMBER_OF_SUBJECTS
# taking last subject ids from shuffled data, so that test subjects of metric learning match the subjects of the classifier
selected_subject_ids = subject_ids[-NUMBER_OF_SUBJECTS:]

output_path = pathlib.Path(OUTPUT_DIR.joinpath(f"/{FPS}_fps_classifier_{len(selected_subject_ids)}_subjects.hdf5"))

if output_path.exists():
    os.remove(output_path)

def write_data(data: pd.DataFrame, key: str):
    data[OFFSET_FRAMES:-CUTOFF_FRAMES].to_hdf(output_path, key=key, mode="a", index=False, dropna=True, append=True, min_itemsize=11)

for session_idx, data in enumerate([session_1_data, session_2_data]):
    data["session_idx"] = session_idx
    data["session_idx"] = data["session_idx"].astype("uint8")

    for subject_id, multi_take_features in data.groupby("subject_id"):
        # there is only one subject that has 2 takes in one session (due to a short interruption during recording),
        # all other subjects have just 1 take
        number_of_takes = len(multi_take_features.subject_take_idx.unique())
        for take_idx, features in multi_take_features.groupby("subject_take_idx"):
            if subject_id in selected_subject_ids:
                if session_idx == 0:
                    # cut off validation data only from the last take of the current session; this requires that the last take is at least
                    # `num_of_frames_in_validation_data` long!
                    if number_of_takes == take_idx + 1:
                        assert len(features) > num_of_frames_in_validation_data
                        # label last quarter as validation and first 3 quarters as train -> equivalent to first 30 minutes and last 10 minutes
                        cut_off_idx_validation_data = len(features) - num_of_frames_in_validation_data
                        train_frames_mask = features['frame_idx'] <= cut_off_idx_validation_data
                        validation_frames_mask = ~train_frames_mask

                        assert train_frames_mask.sum() > 0
                        assert train_frames_mask.sum() > 0

                        write_data(features[train_frames_mask], key="train")
                        write_data(features[validation_frames_mask], key="validation")
                    else:
                        write_data(features, key="train")

                elif session_idx == 1:
                    # all data from session 1 is test data
                    write_data(features, key="test")
                else:
                    raise "unknown session idx!"

print(f"finished, saved data to {output_path}")

# %% Tests

for key in ["train", "validation", "test"]:
    verification_data = pd.read_hdf(output_path, key=key)
    expected_subject_count = NUMBER_OF_SUBJECTS
    actual_subject_count = len(verification_data.subject_id.unique())
    assert expected_subject_count == actual_subject_count, f"unexpected number of subjects found: {actual_subject_count} instead of {expected_subject_count} in {key} set"
    del verification_data

print("all tests passed")
