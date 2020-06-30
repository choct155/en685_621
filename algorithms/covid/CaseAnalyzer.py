import pandas as pd
import numpy as np
from algorithms.covid.CaseStats import CaseStats, CaseStatsByGeo
from typing import List, Sequence
from numbers import Rational

class CaseAnalyzer:

    # def __init__(self, data: pd.Series):
    #     self.data = data

    @staticmethod
    def fold_stats(data: pd.Series) -> List[CaseStats]:

        def loop(data: List[Rational], agg: List[CaseStats]=[]) -> List[CaseStats]:
            if len(data) == 0:
                return agg
            elif len(agg) == 0:
                first: CaseStats = CaseStats(data[0], data[0], data[0], data[0], 0)
                return loop(data[1:], [first])
            else:
                last_stat: CaseStats = agg[-1]
                next_stock: Rational = data[0]
                next_flow: Rational = next_stock - last_stat.stock
                next_max_flow: Rational = next_flow if next_flow > last_stat.max_flow else last_stat.max_flow
                next_onset_days: int = (
                    0 if next_max_flow == 0
                    else last_stat.onset_days + 1
                )
                next_avg_flow: Rational = (
                    0 if next_onset_days == 0
                    else (((next_onset_days - 1) * last_stat.avg_flow) + next_flow) / next_onset_days
                )
                next_stat: CaseStats = CaseStats(
                    next_stock,
                    next_flow,
                    next_max_flow,
                    next_avg_flow,
                    next_onset_days
                )
                return loop(data[1:], agg + [next_stat])
        return loop(data)

    @staticmethod
    def max_mean_by_area(data: pd.DataFrame) -> List[CaseStatsByGeo]:
        geos: Sequence[str] = data.columns
        def get_stats(data: np.array, geo: str) -> CaseStatsByGeo:
            stats: List[CaseStats] = CaseAnalyzer.fold_stats(data)
            last_stat: CaseStats = stats[-1]
            return CaseStatsByGeo(
                geo, 
                last_stat.stock, 
                last_stat.flow, 
                last_stat.max_flow, 
                last_stat.avg_flow, 
                last_stat.onset_days
            )
        stats_by_geo: List[CaseStatsByGeo] = [get_stats(data[geo], geo) for geo in geos]
        return stats_by_geo