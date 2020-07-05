from typing import Sequence, Callable
from numbers import Rational
import numpy as np
import algorithms.utils.CollectionOps as ops

class Likelihood:

    @staticmethod
    def gaussian(mu: Rational, sigma: Rational) -> Callable[[Rational], Rational]:

        def ll(x: Rational) -> Rational:
            base: Rational = 0.5 * np.log( 2 * np.pi * (sigma**2) )
            exp: Rational = 0.5 * ( (x - mu) / (sigma**2) )
            return -1 * (base + exp)
        
        return ll

    @staticmethod
    def log_likelihood(data: Sequence[Rational], dist_ll: Callable[[Rational], Rational]) -> Rational:
        return ops.foldSeq(data, 0, lambda out, next: out + dist_ll(next))