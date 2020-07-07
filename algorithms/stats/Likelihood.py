from typing import Sequence, Callable, List
from numbers import Rational
import numpy as np
import algorithms.utils.CollectionOps as ops
from algorithms.stats.Moment import Moment
from algorithms.data_structures.Vector import Vector

class Likelihood:

    @staticmethod
    def gaussian(mu: Rational, sigma: Rational) -> Callable[[Rational], Rational]:
        """
        Returns a function that takes an observation and returns the log-likelihood
        associated with drawing that observation from a Gaussian distribution 
        parameterized by mu and sigma.
        """
        def ll(x: Rational) -> Rational:
            base: Rational = 0.5 * np.log( 2 * np.pi * (sigma**2) )
            exp: Rational = 0.5 * ( (x - mu)**2 / (sigma**2) )
            return -1 * (base + exp)
        
        return ll

    @staticmethod
    def to_gaussian(data: List[Rational]) -> Callable[[Rational], Rational]: 
        return Likelihood.gaussian(
            Moment.mean(Vector(data)), 
            Moment.std_dev(Vector(data))
        )

    @staticmethod
    def log_likelihood(data: Sequence[Rational], dist_ll: Callable[[Rational], Rational]) -> Rational:
        """ 
        Given a log-likelihood function and data, the function returns the aggregate 
        log-likelihood over the entire data sequence.
        """
        return ops.foldSeq(data, 0, lambda out, next: out + dist_ll(next))