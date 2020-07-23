import tensorflow as tf
from tensorflow.data import Dataset
import numpy as np
from typing import List, Dict, Tuple, Callable, Iterator
from functools import reduce

class IrisPrep:

    @staticmethod
    def combine_data(labels: List[str], raw_data: Dict[str, np.array]) -> Dataset:
        label_data: Callable[[str], Tuple[np.array, np.array]] = lambda lab: (
            raw_data[lab], 
            np.repeat(lab, len(raw_data[lab]))
        )
        data_by_label: Iterator[Tuple[np.array, np.array]] = map(label_data, labels)

        def reduce_grps(first: Tuple[np.array, np.array], second: Tuple[np.array, np.array]) -> Tuple[np.array, np.array]:
            fvals, flabs = first
            svals, slabs = second
            val_out: np.array = np.concatenate([fvals, svals])
            lab_out: np.array = np.concatenate([flabs, slabs])
            return (val_out, lab_out)

        features, labels = reduce(reduce_grps, data_by_label)

        return Dataset.from_tensors((
            features, 
            labels
         ))

    @staticmethod
    def train_test_split(data: Dataset, test_cnt: int) -> Tuple[Dataset, Dataset]:
        values, labels = next(data.as_numpy_iterator()) # both numpy arrays of the same length
        permuted_idxs: np.array  = np.random.permutation(len(values))
        pvals, plabs = (values[permuted_idxs], labels[permuted_idxs])
        train: tf.data.Dataset = Dataset.from_tensors((pvals[test_cnt:], plabs[test_cnt:]))
        test: tf.data.Dataset = Dataset.from_tensors((pvals[:test_cnt], plabs[:test_cnt]))
        return (train, test)