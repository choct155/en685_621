import pytest
import numpy as np
from algorithms.data_structures.Vector import Vector
from typing import Sequence
from numbers import Rational

known_data: Sequence[int] = [1,3,2,5,6,23,2,4,6]

def test_length() -> None:
    v: Vector = Vector(known_data)
    assert v.length == 9

def test_min() -> None:
    v: Vector = Vector(known_data)
    assert v.min() == 1

def test_max() -> None:
    v: Vector = Vector(known_data)
    assert v.max() == 23

def test_append() -> None:
    v: Vector = Vector([1,2,3])
    next_val: int = 4
    test: Vector = v.append(next_val)
    truth: Vector = Vector([1,2,3,4])
    assert test == truth

def test_concat() -> None:
    v1: Vector = Vector([1,2,3])
    v2: Vector = Vector([4,5,6])
    truth: Vector = Vector([1,2,3,4,5,6])
    test: Vector = Vector.concat(v1, v2)
    assert test == truth

def test_filter() -> None:
    input: Vector = Vector([1,2,3,4,5,6])
    truth: Vector = Vector([2,4,6])
    test: Vector = input.filter(lambda x: x % 2 == 0)
    assert test == truth

def test_sort() -> None:
    truth: Vector = Vector([0,1,2,3,4,5,6,7,8,9])
    for i in range(5):
        data: np.array = np.random.permutation(10)
        test: Vector = Vector(data).sort()
        assert test == truth