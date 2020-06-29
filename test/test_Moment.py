import pytest
import numpy as np
from algorithms.data_structures.Vector import Vector
from algorithms.stats.Moment import Moment
from typing import Sequence, Callable
from functools import reduce

simple: Sequence[float] = [1,2,2,4]
n01: Sequence[float] = np.random.normal(loc=0., scale=1., size=100)
n51: Sequence[float] = np.random.normal(loc=5., scale=1., size=100)
n05: Sequence[float] = np.random.normal(loc=0., scale=5., size=100)

vsimple: Vector = Vector(simple)
vn01: Vector = Vector(n01)
vn51: Vector = Vector(n51)
vn05: Vector = Vector(n05)

withinTol: Callable[[float, float], bool] = lambda x, y: (x - y) < 0.01

def test_mean() -> None:
    assert Moment.mean(vsimple) == 2.25
    assert withinTol(Moment.mean(vn01), np.mean(n01)) == True


def test_trimmed_mean() -> None:
    p: int = 10
    assert Moment.trimmed_mean(vsimple, 1) == 2
    assert withinTol(Moment.trimmed_mean(vn01, p), np.mean(n01[p:-p]))

def test_variance() -> None:
    assert withinTol(Moment.variance(vn01), np.var(n01)) == True

def test_std_dev() -> None:
    assert withinTol(Moment.std_dev(vn01), np.std(n01)) == True
