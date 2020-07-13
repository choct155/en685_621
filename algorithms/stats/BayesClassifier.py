from algorithms.stats.ExpectationMaximization import ExpectationMaximization
from numbers import Rational
import numpy as np
from functools import reduce
from typing import List, Dict, Iterable

class BayesClassifier:
    
    def __init__(self, train: np.array, test: np.array, priors: np.array, k: int, tol: float = 0.001) -> None:
        self.train = train
        self.test = test
        self.priors = priors
        self.k = k
        self.tol = tol
        self.stats = None
        self.params = {
            "train": self.train,
            "test": self.test,
            "priors": self.priors,
            "k": self.k,
            "tol": self.tol,
            "stats": self.stats
        }
        
    def fit(self, verbose: bool = False) -> None:
        em: ExpectationMaximization = ExpectationMaximization(self.train, self.k, self.tol)
        em_results: Dict[int, Dict[str,np.array]] = em.fit()
        if verbose:
            print(em_results)
        self.stats = em_results[max(em_results.keys())] # type: ignore
    
    def gaussian(self, obs: np.array, mean: np.array, cov: np.array) -> Rational:
        dim: int = len(obs)
        cov_det: np.array = np.linalg.det(cov)
        cov_inv: np.array = np.linalg.inv(cov)
        base: Rational = 1 / np.sqrt( ((2 * np.pi)**dim) * cov_det )
        # exponential: Rational = np.exp(-0.5 * (obs - mean).T.dot(cov_inv).dot(obs - mean))
        exponential: Rational = np.exp(-0.5 * (obs - mean).T.dot(obs - mean) * (1 / cov_inv))
        return base * exponential
    
    def __classify(self, obs: np.array, means: np.array, covs: np.array, priors: np.array) -> int:
        likelihoods: List[Rational] = list(map(
            lambda idx: priors[idx] * self.gaussian(
                obs, 
                means[:,idx], 
                np.array([[covs[idx]]])
            ), 
            range(self.k)
        ))
        denominator: Rational = reduce(lambda x, y: x+ y,likelihoods)
        scores: Iterable[Rational] = map(lambda s: s/denominator, likelihoods)
        return np.argmax(list(scores))
            
    def classify_obs(self, obs: np.array):
        if self.stats == None:
            self.fit()
        return self.__classify(obs, self.stats["mean"], self.stats["std"], self.priors) # type: ignore (already checked the None case)
    
    def classify_test_data(self) -> np.array:
        return map(lambda obs: self.classify_obs(obs), self.test)