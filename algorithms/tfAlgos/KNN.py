import tensorflow as tf 
from typing import Callable, Tuple, List, Iterator
import numpy as np

class KNN:

    def __init__(self, train: tf.data.Dataset, test: tf.data.Dataset, k: int, labels: List[str]) -> None:
        self.train = train
        self.test = test
        self.k = k
        self.labels = labels

    @staticmethod
    def calc_distance_to_point(p0: tf.Tensor) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        def calc_distance(pn: tf.Tensor, labels: tf.Tensor):
            distances: tf.Tensor = tf.map_fn(lambda xval: tf.norm(p0 - xval, ord="euclidean"), pn)
            return (distances, labels)
        return calc_distance

    # Does it matter if I am feeding in numpy arrays instead of single wrapped tensors?
    @staticmethod
    def calc_distances_to_point(p0: tf.Tensor, data: tf.data.Dataset) -> tf.data.Dataset:
        return data.map(KNN.calc_distance_to_point(p0))

    # Separating this out because I can envision a sliding collection that has better
    # performance than sorting the entire list
    @staticmethod
    def get_max_k(distances: tf.data.Dataset, k: int) -> tf.data.Dataset:
        values, labels = distances.map(lambda x, y: ((1/x), y)).as_numpy_iterator().next()
        top_vals, top_idxs = tf.math.top_k(values, k, sorted=True)
        top_labs = tf.gather(labels, top_idxs)
        return tf.data.Dataset.from_tensors((top_vals, top_labs))

    @staticmethod
    def get_label(max_k: tf.data.Dataset, labels: List[str]) -> str:
        # to bytes: https://stackoverflow.com/questions/6269765/what-does-the-b-character-do-in-front-of-a-string-literal
        cats: List[bytes] = list(map(lambda s: s.encode("UTF-8"), labels))
        values, labels = max_k.as_numpy_iterator().next()
        out = (np.zeros(len(cats)), np.array(cats))
        # effectively reducing and leveraging the input category list
        for i, v in enumerate(values):
            out[0][np.argwhere(out[1] == labels[i])] += v # weights decline with distance because they have been inverted 
        # returning the label with the highest aggregate, distance discounted weight
        return out[1][np.argmax(out[0])]

    def classify_one_obs(self, obs: tf.Tensor) -> str:
        distances: tf.data.Dataset = KNN.calc_distances_to_point(obs, self.train)
        max_k: tf.data.Dataset = KNN.get_max_k(distances, self.k)
        label: str = KNN.get_label(max_k, self.labels)
        return label

    def fit(self) -> tf.data.Dataset:
        test_vals, test_labs = self.test.as_numpy_iterator().next()
        pred_labs = tf.constant(list(map(lambda xval: self.classify_one_obs(xval), test_vals)))
        return tf.data.Dataset.from_tensors((test_vals, test_labs, pred_labs))

    @staticmethod
    def accuracy(fit_data: tf.data.Dataset) -> float:
        test_vals, test_labs, pred_labs = fit_data.as_numpy_iterator().next()
        compare: np.array = np.array([test_labs[i] == pred_labs[i] for i in range(len(pred_labs))])
        return compare.sum() / len(compare)
