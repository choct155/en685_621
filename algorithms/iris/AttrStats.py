from numbers import Rational
import numpy as np
from dataclasses import dataclass

@dataclass
class Stats:
    cov: Rational
    mean: Rational
    min: Rational
    max: Rational 

class AttrStats:

    def __init__(self, data: np.array):
        self.data = data

    def describe(self) -> Stats:
        return Stats(
            np.cov(self.data.T),
            np.mean(self.data, axis=0),
            np.min(self.data, axis=0),
            np.max(self.data, axis=0)
        )
