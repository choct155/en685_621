from algorithms.iris.IrisOps import IrisOps
from typing import Dict, Tuple, Callable, List
import numpy as np
from functools import reduce
import plotly.figure_factory as ff
import plotly.graph_objects as go

class Parzen:
    
    def __init__(
        self, 
        data: Dict[str, np.array], 
        spread: float, 
        train_prop: float = 0.8, 
        labels: List[str] = ["setosa", "versicolor", "virginica"]
    ) -> None:
        self.data = data
        self.spread = spread
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
    def gaussian_kernel(obs: np.array, data: np.array, spread: float) -> np.array:
        obs_rows, obs_cols = obs.shape
        data_rows, data_cols = data.shape
        out: np.array = np.zeros((obs_rows, data_rows))
    
        def g(x_0: np.array, x_n: np.array) -> float:
            normalization: float = 1 / ((np.sqrt(2*np.pi)*spread)**data_cols)
            distance: float = (x_0-x_n).dot((x_0-x_n).T)
            exponential: float = np.exp((-0.5/spread**2) * distance)
            return normalization * exponential

        for i, o in enumerate(obs):
            for j, d in enumerate(data):
                out[i, j] = g(o, d)
            

        return out
    
    @staticmethod
    def score_class(test_obs: np.array, train: np.array, class_idx: int, spread: float) -> float:
        tobs_2d: np.array = test_obs.reshape(1,-1)[:, 1:]
        class_train: np.array = train[train[:, 0] == class_idx]
        kernel_mat: np.array = Parzen.gaussian_kernel(tobs_2d, class_train[:, 1:], spread)
        return kernel_mat.sum()
    
    @staticmethod
    def label_obs(test_obs: np.array, train: np.array, spread: float) -> int:
        labels: np.array = np.unique(train[:, 0])
        scores: List[Tuple[int, np.array]] = list(map(
            lambda class_idx: (class_idx, Parzen.score_class(test_obs, train, class_idx, spread)),
            labels
        ))
        max_score: Tuple[int, np.array] = reduce(lambda f, s: f if f[1] >= s[1] else s, scores)
        return max_score[0]
    
    def fit(self) -> np.array:
        truth: np.array = self.test[:, 0].reshape(len(self.test), 1)
        pred: np.array = np.array(list(
            map(lambda obs: Parzen.label_obs(obs, self.train, self.spread), self.test)
        )).reshape(len(self.test), 1)
        return np.concatenate([truth, pred], axis=1)
        
    @staticmethod
    def accuracy(label_pairs: np.array) -> float:
        matches: int = len(label_pairs[label_pairs[:, 0] == label_pairs[:, 1]])
        total: int = len(label_pairs)
        return matches / total
    
    @staticmethod
    def plot1D(
        raw_data: Dict[str,np.array], 
        support: np.array, 
        feature: int, 
        spread: float,
        labels: List[str],
        color_map: Dict[str, str]
    ) -> go.Figure:
        
        input_data: List[np.array] = [raw_data[lab][:,feature] for lab in labels]
        
        pgk: Callable[[float], float] = lambda obs: Parzen.gaussian_kernel(
            np.array([obs]).reshape(1,1), 
            support.reshape(len(support), 1), 
            spread
        )
        def class_density(feature_arr: np.array) -> np.array:
            arrs: List[np.array] = list(map(lambda obs: pgk(obs), feature_arr))
            sum_array: np.array = reduce(lambda f, s: f + s, arrs)
            return sum_array / len(feature_arr)
    
        hist_data: List[np.array] = [class_density(arr)[0] for arr in input_data]
        colors: List[str] = [color_map[lab] for lab in labels]
        
        fig: go.Figure = go.Figure()
        for i, lab in enumerate(labels):
            fig.add_trace(go.Scatter(
                x = support,
                y = hist_data[i],
                line=dict(color=colors[i]),
                    name=f"{lab} density"
            ))
            fig.add_trace(go.Scatter(
                x = input_data[i],
                y = np.random.uniform(-0.05, 0.05, size = len(input_data[i])),
                mode = "markers",
                opacity = 0.7,
                marker = dict(
                    color="#ffffff", 
                    line = dict(color=colors[i], width=1)
                ),
                name = f"{lab} data"
            ))
        fig.update_layout(template="plotly_white")
        return fig
    
    @staticmethod
    def plot2D(
        raw_data: Dict[str,np.array], 
        x_support: np.array, 
        y_support: np.array, 
        features: List[int], 
        spread: float,
        labels: List[str],
        color_map: Dict[str, str]
    ) -> go.Figure:
        
        input_data: List[np.array] = [raw_data[lab][:,features] for lab in labels]
        
        def gk_grid(obs: np.array, x_support: np.array, y_support: np.array, spread: float) -> np.array:
            out: np.array = np.zeros((len(x_support), len(y_support)))
                
            for i, x in enumerate(x_support):
                for j, y in enumerate(y_support):
                    pgk: float = Parzen.gaussian_kernel(
                        np.array([obs]).reshape(1,2),
                        np.array([[x, y]]),
                        spread
                    )
                    # print(pgk)
                    out[i,j] = pgk
            return out
        
        def class_density(feature_arr: np.array, x_support: np.array, y_support: np.array, spread: float) -> np.array:
            arrs: List[np.array] = list(map(lambda obs: gk_grid(obs, x_support, y_support, spread), feature_arr))
            sum_array: np.array = reduce(lambda f, s: f + s, arrs)
            return sum_array / len(feature_arr)
    
        contour_data: List[np.array] = [class_density(arr, x_support, y_support, spread)[0] for arr in input_data]
        consolidated_contour: np.array = reduce(lambda f, s: f + s, contour_data)
        colors: List[str] = [color_map[lab] for lab in labels]
        contour_colors: List[str] = ["Reds", "Blues", "Greens"]
        
        fig: go.Figure = go.Figure()
        # fig.add_trace(go.Contour(
        #     z = contour_data[0],
        #     x = x_support,
        #     y = y_support
        # ))
        for i, lab in enumerate(labels):
            fig.add_trace(go.Histogram2dContour(
                x = [input_data[i][j][0] for j in range(len(input_data[i]))],
                y = [input_data[i][j][1] for j in range(len(input_data[i]))],
                colorscale = contour_colors[i]
            ))
            fig.add_trace(go.Scatter(
                x = [input_data[i][j][0] for j in range(len(input_data[i]))],
                y = [input_data[i][j][1] for j in range(len(input_data[i]))],
                mode = "markers",
                opacity = 0.7,
                marker = dict(
                    color="#ffffff", 
                    line = dict(color=colors[i], width=3)
                ),
                name = f"{lab} data"
            ))

        fig.update_layout(template="plotly_white")
        return fig
    
    