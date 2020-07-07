from typing import List, Callable, Tuple
from numbers import Rational
from dataclasses import dataclass
from algorithms.stats.Moment import Moment
from algorithms.data_structures.Vector import Vector
import algorithms.utils.CollectionOps as ops
import numpy as np

@dataclass
class ConfidenceIntervalResults:
    mean: Rational
    std_upper: Rational
    std_lower: Rational
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

        return ConfidenceIntervalResults(
            global_mean,
            upper_std_dev,
            lower_std_dev,
            upper_data + lower_data, 
            Vector(lower_outliers).sort().data, 
            Vector(upper_outliers).sort().data
        )

    def remove_n_outliers(self, n: int) -> List[Rational]:
        ci: ConfidenceIntervalResults = self.byConfidenceInterval()
        dev: Callable[[Rational], Tuple[Rational, Rational]] = lambda obs: (obs, abs(ci.mean - obs))
        # TODO: Cheating here with a built in
        outlier_deviations: List[Tuple[Rational, Rational]] = sorted(
            list(map(dev, ci.upper_outliers + ci.lower_outliers)), 
            key=lambda tup: tup[1]
        )
        with_n_removed: List[Rational] = list(map(lambda tup: tup[0], outlier_deviations[:-n])) if n < len(outlier_deviations) else []

        return ci.out_data + with_n_removed

        