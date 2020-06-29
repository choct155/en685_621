from typing import Sequence
from numbers import Rational
from algorithms.data_structures.Vector import Vector
from functools import reduce
import numpy as np
import pandas as pd

class Matrix:

    def __init__(self, _data: np.array) -> None:
        self.data: np.array = _data
        self.shape: int = np.array(_data).shape



def fromVecSeq(vecs: Sequence[Vector]) -> Matrix:
    if len(vecs) <= 1:
        raise ValueError("Input vector sequence must be of positive length")
    elif reduce(lambda v1, v2: v1.length == v2.length, vecs):
        raise ValueError("All input vectors must be of the same length")
    else: 
        data: np.array = np.array([vec.data for vec in vecs])
        return Matrix(data)

def fromCsv(path: str) -> Matrix:
    data: np.array = pd.read_csv(path).values
    return Matrix(data)

