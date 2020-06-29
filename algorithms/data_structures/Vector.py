from typing import TypeVar, Sequence, Callable, Optional, Tuple, Iterator
from functools import reduce
from numbers import Rational
import algorithms.utils.CollectionOps as ops

class Vector:

    def __init__(self, _data: Sequence[Rational]) -> None:
        if len(_data) <= 1:
            raise ValueError("Input sequence must be of positive length")
        else: 
            self.data: Sequence[Rational] = _data
            self.length = len(_data)
            self.index = 0

    def __iter__(self):
        self.index: int = 0
        return self

    def __next__(self):
        if self.index < self.length:
            elem: Rational = self.data[self.index]
            self.index += 1
            return elem
        else:
            raise StopIteration

    def min(self) -> Rational:
        return reduce(lambda x, y: x if x < y else y, self.data)

    def max(self) -> Rational:
        return reduce(lambda x, y: x if x > y else y, self.data)
