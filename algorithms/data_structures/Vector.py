from typing import TypeVar, List, Callable, Optional, Tuple, Iterator
from functools import reduce
from numbers import Rational
import algorithms.utils.CollectionOps as ops
import algorithms.data_structures.BinaryTree as bt

class BaseVector:

    def __init__(self, _data: List[Rational]) -> None:
        if len(_data) <= 1:
            raise ValueError("Input sequence must be of positive length")
        else: 
            self.data: List[Rational] = _data
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

    
class Vector(BaseVector):

    def __init__(self, _data: List[Rational]):
        super().__init__(_data)

    # HT: https://stackoverflow.com/questions/37557411/why-does-defining-the-argument-types-for-eq-throw-a-mypy-type-error
    def __eq__(self, that: object) -> bool:
        if not isinstance(that, Vector):
            return NotImplemented
        elif len(self.data) != len(that.data):
            return False
        else:
            comparisons: List[bool] = [self.data[i] == that.data[i] for i in range(len(self.data))]
            return reduce(lambda first, second: first & second, comparisons)

    def __add__(self, that: object) -> Optional[object]:
        if not isinstance(that, Vector):
            return NotImplemented
        elif len(self.data) != len(that.data):
            return None
        else:
            new_data: List[Rational] = ops.combineListElems(self.data, that.data, lambda s, t: s + t)
            return Vector(new_data)


    def min(self) -> Rational:
        return reduce(lambda x, y: x if x < y else y, self.data)

    def max(self) -> Rational:
        return reduce(lambda x, y: x if x > y else y, self.data)

    @staticmethod
    def empty() -> BaseVector:
        return Vector([])

    def append(self, value: Rational) -> BaseVector:
        new_data: List[Rational] = self.data + [value]
        return Vector(new_data)

    def concat(self, that: BaseVector) -> BaseVector:
        new_data: List[Rational] = ops.concatLists(self.data, that.data)
        return Vector(new_data)

    def filter(self, predicate: Callable[[Rational], bool]) -> BaseVector:
        new_data: List[Rational] = ops.filterList(self.data, predicate)
        return Vector(new_data)

    def sort(self) -> BaseVector:
        tree: bt.NonEmptyNode = bt.NonEmptyNode.init_tree(self.data)
        sorted_data: List[Rational] = tree.sorted_values()
        return Vector(sorted_data)


