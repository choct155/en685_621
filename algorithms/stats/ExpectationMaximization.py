import numpy as np
from dataclasses import dataclass
from numbers import Rational
from typing import Dict, Optional

@dataclass
class InitParams:
    mean: Rational
    std: Rational

@dataclass
class IterParams:
    mean: Rational
    std: Rational
    weight: Rational
    iteration: int

class ExpectationMaximization:

    def __init__(self, data: np.array, k: int, tol: float = 0.001) -> None:
        self.init_data = data
        self.k = k
        self.n_rows = data.shape[0]
        self.n_cols = data.shape[1]
        self.init_mean = self.__init_mean()
        self.init_std = self.__init_std()
        self.init_p = self.__init_p()
        self.tol = tol

    def __init_mean(self) -> np.array:
        first_mean: np.array = np.outer(self.init_data.mean(axis=0),np.ones(self.k))
        first_std: np.array = np.outer(self.init_data.std(axis=0), np.random.uniform(-1,1,size=self.k))
#         first_std: np.array = np.outer(self.init_data.std(axis=0, ddof=1), np.ones(self.k))
        return first_mean + first_std

    def __init_std(self) -> np.array:
        avg_std: Rational = self.init_data.std(axis=0, ddof=1).mean()
        return avg_std * np.ones(self.k)

    def __init_p(self) -> np.array:
        return np.ones(self.k) / self.k

    def gaussian(self, std: Rational, obs: np.array, mean: np.array) -> Rational:
        base: Rational = (np.sqrt(2 * np.pi) * std)**self.n_cols
        exponent: Rational = -0.5 * (((obs - mean).dot(obs - mean)) / (std**2))
        return (1/base)*np.exp(exponent)

    def expectation(self, mean: np.array, std: np.array, p: np.array) -> np.array:
        out: np.array = np.zeros((self.k, self.n_rows))

        for i, obs in enumerate(self.init_data):
            for grp in range(self.k):
                obs_g: Rational = self.gaussian(std[grp], obs, mean[:, grp])
                obs_p: Rational = p[grp] * obs_g
                out[grp, i] = obs_p

        return out / out.sum(axis=0)

    def update_mean(self, rs: np.array) -> np.array:
        out: np.array = np.zeros((self.n_cols, self.k))
        
        for grp in range(self.k):
            new_col: np.array = self.init_data.T.dot(rs[grp]) / rs[grp].sum()
            out[:, grp] = new_col
        
        return out
    
    def update_std(self, rs: np.array, mean: np.array) -> np.array:
        out: np.array = np.ones(self.k)
        
        for grp in range(self.k):
            squared_deviations: np.array = ((self.init_data - mean[:,grp])**2).sum(axis=1)
            weighted_sd: Rational = squared_deviations.dot(rs[grp])
            responsibilities: Rational = self.k * rs[grp].sum()
            out[grp] = np.sqrt(weighted_sd/responsibilities)
        
        return out
    
    def update_weights(self, rs: np.array) -> np.array:
        return rs.sum(axis=1) / self.n_rows
    
    @staticmethod
    def tolerance(t: float, arr1: np.array, arr2: np.array) -> bool:
        norm1: np.array = arr1 / np.linalg.norm(arr1)
        norm2: np.array = arr2 / np.linalg.norm(arr2)
        dot_prod: Rational = norm1.dot(norm2)
        return abs(dot_prod - 1) < t
    
    def fit(self, num_iter: Optional[int] = None) -> Dict[int, Dict[str,np.array]]:
        initial_responsibilities: np.array = self.expectation(self.init_mean, self.init_std, self.init_p)
        out: Dict[int, Dict[str,np.array]] = {
            0: {
                "responsibilities": initial_responsibilities,
                "mean": self.init_mean,
                "std": self.init_std,
                "weights": self.init_p
            }
        }

        if num_iter != None:
            for i in range(num_iter): # type: ignore already checked the None case
                updated_mean: np.array = self.update_mean(out[i]["responsibilities"])
                updated_std: np.array = self.update_std(out[i]["responsibilities"], updated_mean)
                updated_p: np.array = self.update_weights(out[i]["responsibilities"])
                updated_responsibilities: np.array = self.expectation(updated_mean, updated_std, updated_p)
            
                iter_dict = {
                    "responsibilities": updated_responsibilities,
                    "mean": updated_mean,
                    "std": updated_std,
                    "weights": updated_p
                }
            
                out.update({i + 1: iter_dict})
                
        else:
            
            converged: bool = False
            idx: int = 0
        
            while not converged:
                conv_updated_mean: np.array = self.update_mean(out[idx]["responsibilities"])
                conv_updated_std: np.array = self.update_std(out[idx]["responsibilities"], conv_updated_mean)
                conv_updated_p: np.array = self.update_weights(out[idx]["responsibilities"])
                conv_updated_responsibilities: np.array = self.expectation(conv_updated_mean, conv_updated_std, conv_updated_p)

                iter_dict = {
                    "responsibilities": conv_updated_responsibilities,
                    "mean": conv_updated_mean,
                    "std": conv_updated_std,
                    "weights": conv_updated_p
                }

                converged = ExpectationMaximization.tolerance(self.tol, out[idx]["weights"], conv_updated_p)
                idx += 1
                out.update({idx: iter_dict})
            
        return out
            
            
        
        
