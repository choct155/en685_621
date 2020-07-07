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

    data_good: List[bool] = list(map(lambda obs: (obs <= upper) & (obs >= lower), ci.out_data))
    upper_good: List[bool] = list(map(lambda obs: obs > upper, ci.upper_outliers))
    lower_good: List[bool] = list(map(lambda obs: obs < lower, ci.lower_outliers))

    data_good_check = list(map(lambda obs: (obs, (obs <= upper) & (obs >= lower)), ci.out_data))
    for pair in data_good_check:
        assert (pair[1] == True) & isinstance(pair[0], float)

    assert (len(ci.out_data) > 0) & (np.all(list(data_good)))
    assert (len(ci.upper_outliers) > 0) & (np.all(list(upper_good)))
    assert (len(ci.lower_outliers) > 0) & (np.all(list(lower_good)))