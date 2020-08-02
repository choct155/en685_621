import numpy as np
from sklearn.svm import SVC
from algorithms.iris.IrisOps import IrisOps
from typing import Dict, Tuple, Callable, List
from functools import reduce

class SupportVector:
    
    def __init__(
        self, 
        data: Dict[str, np.array], 
        meshdim: Tuple[int, int] = (50, 50),
        train_prop: float = 0.8, 
        labels: List[str] = ["setosa", "versicolor", "virginica"]
    ) -> None:
        self.data = data
        self.meshdim = meshdim
        self.train_prop = train_prop
        self.labels = labels
        self.train, self.test = self.split()
        
    def split(self) -> Tuple[np.array, np.array]:
        return IrisOps.test_train_split(self.data, self.labels, self.train_prop)
        
    def preprocess(self, data: np.array) -> Tuple[np.array, np.array]:
        y: np.array = data[:,0]
        x_in: np.array = data[:, 1:]
        X: np.array = (x_in - x_in.mean(axis=0)) / x_in.std(axis=0)
        return np.concatenate([y, X], axis=1)
    
    @staticmethod
    def fit(features: List[int], train: np.array, kernel: str = "rbf", C: int = 1, gamma: float = 0.5) -> SVC:
        X, y = (train[:, features], train[:, 0])
        return SVC(kernel=kernel, gamma=gamma, C=C).fit(X, y)
    
    @staticmethod
    def accuracy(clf: SVC, test: np.array, features: List[int]) -> Tuple[np.array, np.array]:
        truth: np.array = test[:, 0].reshape(len(test), 1)
        pred: np.array = clf.predict(test[:, features]).reshape(len(test), 1)
        paired: np.array = np.concatenate([truth, pred], axis=1)
        matches: int = len(paired[paired[:,0] == paired[:,1]])
        total: int = len(paired)
        return (matches / total, paired)
    
    @staticmethod
    def binary_fit(
        class_idx: int,
        features: List[int],
        train: np.array,
        kernel: str = "rbf",
        C: int = 1,
        gamma: float = 0.5
    ) -> SVC:
        X, y_in = (train[:, features], train[:, 0])
        y = np.where(y_in == class_idx, 1, -1)
        return SVC(kernel=kernel, gamma=gamma, C=C).fit(X, y)
    
    @staticmethod
    def binary_round_robin(
        labels: List[str],
        features: List[int],
        train: np.array,
        test: np.array,
        kernel: str = "rbf",
        C: int = 1,
        gamma: float = 0.5
    ) -> SVC:
        
        truth = test[:, 0]
        
        classifiers: Dict[str, SVC] = {
            lab:SupportVector.binary_fit(i, features, train) 
            for i,lab in enumerate(labels)
        }
        
        test_out: Dict[str, np.array] = {
            lab:SupportVector.accuracy(classifiers[lab], test, features)[1]
            for lab in labels
        }
        
        def bin_label(truth_pred: np.array, label: str) -> np.array:
            return np.array([label]) if truth_pred[1] == 1 else np.array(["other"])
        
        pred_dict: Dict[str, np.array] = {
            lab:np.concatenate(list(map(lambda arr: bin_label(arr, lab), test_out[lab])), axis=0) 
            for lab in labels
        }
            
        pred_lab_arrs: np.array = np.concatenate([arr.reshape(len(arr), 1) for arr in pred_dict.values()],axis=1)
        
        pred_labs: List[np.array] = list(map(lambda arr: arr[arr != "other"], pred_lab_arrs))
        
        truth_labs: List[np.array] = list(map(lambda arr: np.array([labels[int(arr)]]), truth))
        
        matches: np.array = np.concatenate(
            [(pred_labs[i] == truth_labs[i]).reshape(1,1) for i in range(len(pred_labs))],
            axis=0
        )
        return dict(accuracy=(len(matches)/len(pred_labs)), predictions=pred_labs, truth=truth_labs)
        
            