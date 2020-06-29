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
