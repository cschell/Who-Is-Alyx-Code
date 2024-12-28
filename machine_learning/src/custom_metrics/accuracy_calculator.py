from collections import Counter
from typing import Union, List

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=(len(arr), bins), dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    np.add.at(count, (indexing, arr), 1)

    return count


class MotionAccuracyCalculator(AccuracyCalculator):
    def __init__(self, sequence_lengths_minutes: List[int], sliding_window_step_size_seconds: int, max_k_for_majority_vote=50, step_size=10, query_fps=15, **kwargs):
        self.query_fps = query_fps
        self.i_range = np.append(0, np.arange(step_size - 1, max_k_for_majority_vote, step_size))
        self.max_k_for_majority_vote = max_k_for_majority_vote
        for i in self.i_range:
            setattr(self, f"calculate_majority_vote_at_{i + 1}", lambda knn_labels, query_labels, i=i, **kwargs: self.majority_vote(i, knn_labels, query_labels, **kwargs))

        self.sliding_window_step_size_seconds = sliding_window_step_size_seconds
        self.sequence_lengths_minutes = sequence_lengths_minutes
        for seq_length in self.sequence_lengths_minutes:
            setattr(self, f"calculate_sequence_mrr_at_{seq_length}_mins",
                    lambda knn_labels, query_labels, seq_length=seq_length, **kwargs: self.sequence_calculations_cache[seq_length]["mrr"])
            setattr(self, f"calculate_sequence_top_1_accuracy_{seq_length}_mins",
                    lambda knn_labels, query_labels, seq_length=seq_length, **kwargs: self.sequence_calculations_cache[seq_length]["top_1_accuracy"])
            setattr(self, f"calculate_sequence_top_2_accuracy_{seq_length}_mins",
                    lambda knn_labels, query_labels, seq_length=seq_length, **kwargs: self.sequence_calculations_cache[seq_length]["top_2_accuracy"])
            setattr(self, f"calculate_sequence_top_3_accuracy_{seq_length}_mins",
                    lambda knn_labels, query_labels, seq_length=seq_length, **kwargs: self.sequence_calculations_cache[seq_length]["top_3_accuracy"])
        super().__init__(**kwargs)

    def majority_vote(self, i, knn_labels: Union[torch.Tensor, np.ndarray], query_labels: Union[torch.Tensor, np.ndarray], **kwargs):
        if type(knn_labels) == torch.Tensor:
            knn_labels = knn_labels.detach().cpu().numpy()

        if type(query_labels) == torch.Tensor:
            query_labels = query_labels.detach().cpu().numpy()

        correct = 0
        for query_label, knn_label in zip(query_labels, knn_labels):
            top_i = knn_label[:i + 1]
            most_common = Counter(top_i).most_common(1)[0][0]

            if most_common == query_label:
                correct += 1

        return correct / len(query_labels)

    def _get_accuracy(self, function_dict, **kwargs):
        self.sequence_calculations_cache = {}
        for minutes in self.sequence_lengths_minutes:
            self._perform_sequence_calculations(minutes, **kwargs)

        return super()._get_accuracy(function_dict, **kwargs)

    def _perform_sequence_calculations(self, sequence_length_minutes, knn_labels, query_labels, **kwargs):
        assert np.all(np.sort(np.unique(query_labels)) == np.arange(query_labels.max() + 1))

        sliding_window_step_size_frames = int(np.round(self.sliding_window_step_size_seconds * self.query_fps))
        window_size = int(np.round(sequence_length_minutes * 60 * self.query_fps))
        knn_majority_vote_at = 50

        klasses = np.unique(query_labels).astype(int)
        num_klasses = len(klasses)

        per_sample_neighbour_counts = bincount2d(knn_labels[:, :knn_majority_vote_at].numpy().astype(int)) / knn_majority_vote_at

        mrrs = []
        top_1_ranks = []
        top_2_ranks = []
        top_3_ranks = []
        for klass in klasses:
            mask = query_labels == klass

            slided_window_view = sliding_window_view(per_sample_neighbour_counts[mask], (window_size, num_klasses))[::sliding_window_step_size_frames].squeeze()

            per_class_count = np.sum(slided_window_view, axis=1) / window_size

            ranks = (num_klasses - per_class_count.argsort().argsort())[:, klass]

            top_1_ranks.append((ranks == 1).mean())
            top_2_ranks.append((ranks <= 2).mean())
            top_3_ranks.append((ranks <= 3).mean())
            reciprocal_ranks = 1 / ranks
            mrr = np.mean(reciprocal_ranks)
            mrrs.append(mrr)

        self.sequence_calculations_cache[sequence_length_minutes] = {
            "mrr": np.mean(mrrs),
            "top_1_accuracy": np.mean(top_1_ranks),
            "top_2_accuracy": np.mean(top_2_ranks),
            "top_3_accuracy": np.mean(top_3_ranks),
        }

    def requires_knn(self):
        return super().requires_knn() + \
               [f"sequence_mrr_at_{seq_length}_mins" for seq_length in self.sequence_lengths_minutes] + \
               [f"majority_vote_at_{i + 1}" for i in self.i_range]
