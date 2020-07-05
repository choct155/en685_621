import numpy as np
from typing import TypeVar, Sequence, Callable, Optional, Tuple, Iterator, List
from functools import reduce
from numbers import Rational
import algorithms.utils.CollectionOps as ops
from algorithms.data_structures.Vector import Vector
from algorithms.data_structures.Matrix import Matrix

N = TypeVar('N', int, float)

class Moment:

    @staticmethod
    def mean(data: Vector) -> Rational:
        return reduce(lambda x, y: x + y, data) / data.length

    @staticmethod
    def trimmed_mean(data: Vector, p: int) -> Rational:
        trimmed_data: Vector = Vector(data.data[p:-p])
        return Moment.mean(trimmed_data)

    @staticmethod
    def variance(data: Vector) -> Rational:
        mu: Rational = Moment.mean(data)
        sum_squares: Rational = ops.foldSeq(data, 0, lambda out, xi: out + (xi - mu)**2)
        return sum_squares / (data.length - 1)

    @staticmethod
    def std_dev(data: Vector) -> Rational:
        return Moment.variance(data)**0.5

    @staticmethod
    def covariance(this: Vector, that: Vector) -> Rational:
        if this.length != that.length:
            raise ValueError("Input vectors must be of equivalent length")
        zipped: Iterator[Tuple[Rational, Rational]] = zip(this.data, that.data)
        dev_product: Callable[[Rational, Rational], Rational] = lambda s, o: (s - Moment.mean(this)) * (o - Moment.mean(that))
        sum_dev_prods: Rational = ops.foldSeq(zipped, 0, lambda out, next: out + dev_product(next[0], next[1]))
        return sum_dev_prods / (this.length - 1)

    @staticmethod
    def higherOrderMoment(data: Vector, order: int) -> Rational:
        sum_exp_dev: Rational = ops.foldSeq(data, 0, lambda out, next: out + (next - Moment.mean(data))**order)
        raw_moment: Rational = sum_exp_dev / (data.length - 1)
        return raw_moment / (Moment.std_dev(data)**order)

    @staticmethod
    def skewness(data: Vector) -> Rational:
        return Moment.higherOrderMoment(data, 3)

    @staticmethod
    def kurtosis(data: Vector) -> Rational:
        return Moment.higherOrderMoment(data, 4)

    @staticmethod
    def covariance_matrix(mat: Matrix) -> Matrix:
        data: np.array = np.cov(mat.data)
        return Matrix(data)