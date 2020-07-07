from algorithms.stats.Moment import Moment
from algorithms.data_structures.Vector import Vector
from algorithms.stats.Likelihood import Likelihood
import algorithms.utils.CollectionOps as ops
from numbers import Rational
from typing import List, Callable, Tuple
import numpy as np

class StatDistance:

    @staticmethod
    def gaussian_misclassification(this: List[Rational], that: List[Rational]) -> Rational:
        """ 
        Calculates the sample means and standard deviations of this and that, and then uses
        them to calculate the log-likelihood of drawing this from that and vice versa. The
        Results are then averaged together to provide a relative measure of the likelihood of
        misclassification. Lower values indicate lower probability of misclassification.

        Since likelihood scales with the number of observations, the same number of observations
        are used from both this and that.
        """
        this_g: Callable[[Rational], Rational] = Likelihood.to_gaussian(this)
        that_g: Callable[[Rational], Rational] = Likelihood.to_gaussian(that)

        # permuting in case we get a sorted list
        zipped: List[Tuple[Rational, Rational]] = ops.zipLists(np.random.permutation(this), np.random.permutation(that))

        def tuple_ll_f(agg: Tuple[Rational, Rational], next_val: Tuple[Rational, Rational]) -> Tuple[Rational, Rational]:
            """ 
            Combines likelihoods of drawing the next value of this or that from the other distribution with
            the running likelihood total.
            """
            this_agg, that_agg = agg
            this_next, that_next = next_val
            return (this_agg + that_g(this_next), that_agg + this_g(that_next))

        zipped_ll: Tuple[Rational, Rational] = ops.foldSeq(zipped, (0,0), tuple_ll_f)
        this_ll, that_ll = zipped_ll

        return (this_ll + that_ll) / 2

    @staticmethod
    def gaussian_misclassification_matrix(data: np.array) -> np.array:
        """
        Intended for use when comparing a collection of data series, for which we need to know
        the misclassification likelihoods. Each column of the input data is a data series, and
        each row (i) captures the ith observation of each data series. The function returns an 
        square array that has a number of columns and rows that is equal to the number of
        columns in the input data.
        """
        nrows, ncols = data.shape
        out: np.array = np.zeros((ncols, ncols)) # don't love the upcoming mutation, but it's contained
        for col1 in range(ncols):
            for col2 in range(ncols):
                out[col1][col2] = StatDistance.gaussian_misclassification(
                    data[:,col1], 
                    data[:,col2]
                )

        return out