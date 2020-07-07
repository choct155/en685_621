import numpy as np
from typing import Callable, List
from numbers import Rational
from algorithms.stats.Likelihood import Likelihood
from algorithms.data_structures.Vector import Vector
from algorithms.stats.Moment import Moment

def test_gaussian() -> None:
    truth: np.array = np.random.normal(0, 1, size=100)
    wide: np.array = np.random.normal(0, 3, size=100)
    biased: np.array = np.random.normal(6, 1, size=100)

    to_ll: Callable[[List[Rational]], Callable[[Rational], Rational]] = lambda data: Likelihood.gaussian(
        Moment.mean(Vector(data)), Moment.std_dev(Vector(data))
    )

    truth_ll: Callable[[Rational], Rational] = to_ll(truth)
    wide_ll: Callable[[Rational], Rational] = to_ll(wide)
    biased_ll: Callable[[Rational], Rational] = to_ll(biased)

    draw_truth_from_truth: Rational = Likelihood.log_likelihood(truth, truth_ll)
    draw_wide_from_truth: Rational = Likelihood.log_likelihood(wide, truth_ll)
    draw_biased_from_truth: Rational = Likelihood.log_likelihood(biased, truth_ll)

    draw_truth_from_wide: Rational = Likelihood.log_likelihood(truth, wide_ll)
    draw_wide_from_wide: Rational = Likelihood.log_likelihood(wide, wide_ll)
    draw_biased_from_wide: Rational = Likelihood.log_likelihood(biased, wide_ll)

    draw_truth_from_biased: Rational = Likelihood.log_likelihood(truth, biased_ll)
    draw_wide_from_biased: Rational = Likelihood.log_likelihood(wide, biased_ll)
    draw_biased_from_biased: Rational = Likelihood.log_likelihood(biased, biased_ll)

    assert draw_truth_from_truth > draw_wide_from_truth
    assert draw_truth_from_truth > draw_biased_from_truth

    assert draw_wide_from_wide < draw_truth_from_wide # Note that draws from the middle of wide will have higher likelihood
    assert draw_wide_from_wide > draw_biased_from_wide

    assert draw_biased_from_biased > draw_truth_from_biased
    assert draw_biased_from_biased > draw_wide_from_biased