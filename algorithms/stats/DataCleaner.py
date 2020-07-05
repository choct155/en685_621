from typing import List
from numbers import Rational
from dataclasses import dataclass
from algorithms.stats.Moment import Moment
from algorithms.data_structures.Vector import Vector
import algorithms.utils.CollectionOps as ops
import numpy as np

@dataclass
class ConfidenceIntervalResults:
    out_data: List[Rational]
    lower_outliers: List[Rational]
    upper_outliers: List[Rational]

class OutlierDetector:

    def __init__(self, data: List[Rational]) -> None:
        self.data = Vector(data)

    def byConfidenceInterval(self) -> ConfidenceIntervalResults:
        """
        Classifies individual observations as data, lower outliers, and
        upper outliers. Leverages a definition of confidence interval
        that extends one standard deviation of the upper data above,
        and one standard deviation of the lower data below. All other
        data are outliers.
        """
        global_mean: Rational = Moment.mean(self.data)

        upper, lower = ops.splitList(self.data.data, lambda obs: obs <= global_mean)
        upper_std_dev: Rational = Moment.std_dev(Vector(upper))
        lower_std_dev: Rational = Moment.std_dev(Vector(lower))
        np_upper = np.std(upper)
        np_lower = np.std(lower)

        upper_outliers, upper_data = ops.splitList(upper, lambda obs: obs <= global_mean + upper_std_dev)
        lower_outliers, lower_data = ops.splitList(lower, lambda obs: obs >= global_mean - lower_std_dev)

        return ConfidenceIntervalResults(upper_data + lower_data, lower_outliers, upper_outliers)

        