import algorithms.stats.DataCleaner as dc
import numpy as np 
from typing import Iterator, List, Tuple

def test_confidence_interval() -> None:
    n12: np.array = np.random.normal(1, 2, size=100)
    mean: float = np.mean(n12)
    upper_data: np.array = n12[np.where(n12 >= mean)]
    lower_data: np.array = n12[np.where(n12 < mean)]
    upper: float = mean + np.std(upper_data)
    lower: float = mean - np.std(lower_data)

    od: dc.OutlierDetector = dc.OutlierDetector(n12)
    ci: dc.ConfidenceIntervalResults = od.byConfidenceInterval()

    data_good: Iterator[bool] = map(lambda obs: (obs <= upper) & (obs >= lower), ci.out_data) 
    upper_good: Iterator[bool] = map(lambda obs: obs > upper, ci.upper_outliers)
    lower_good: Iterator[bool] = map(lambda obs: obs < lower, ci.lower_outliers)

    assert (len(ci.out_data) > 0) & (np.all(list(data_good)))
    assert (len(ci.upper_outliers) > 0) & (np.all(list(upper_good)))
    assert (len(ci.lower_outliers) > 0) & (np.all(list(lower_good)))