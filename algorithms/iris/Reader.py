import numpy as np
from sklearn import datasets

class IrisReader:

    def __init__(self):
        super().__init__()

    def load(self):
        iris_in: np.array = datasets.load_iris()
        self.data = dict(
            setosa = iris_in["data"][:50],
            versicolor = iris_in["data"][50:100],
            virginica = iris_in["data"][100:]
        )