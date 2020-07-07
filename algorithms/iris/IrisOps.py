import numpy as np
import pandas as pd
from typing import Dict, Tuple, Callable
from numbers import Rational
from algorithms.iris.DataGenerator import DataGenerator
from algorithms.data_structures.Vector import Vector
import plotly.graph_objects as go

class IrisOps:

    @staticmethod
    def getCol(data: np.array, feature: str) -> np.array:
        label_map: Dict[str, int] = dict(sepal_length=0, sepal_width=1, petal_length=2, petal_width=3)
        return data[:, label_map[feature]]

    @staticmethod
    def compare_synth_data(data: Dict[str, np.array], species: str, features: Tuple[str, str]) -> None:
        input_data: np.array = data[species]
        synth_data: np.array = DataGenerator.gen_synthetic_data(input_data)

        fig = go.Figure()
    
        fig.add_trace(
            go.Scatter(
                x=IrisOps.getCol(input_data, features[0]),
                y=IrisOps.getCol(input_data, features[1]),
                mode="markers",
                name="observed"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=IrisOps.getCol(synth_data, features[0]),
                y=IrisOps.getCol(synth_data, features[1]),
                mode="markers",
                name="generated"
            )
        )
        fig.update_layout(
            title="Comparison of Observed and Generated Data",
            template='plotly_white'
        )
        fig.show()

    @staticmethod
    def arrange_by_field(raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        sepal_length: Callable[[np.array], np.array] = lambda data: data[:,0]
        sepal_width: Callable[[np.array], np.array] = lambda data: data[:,1]
        petal_length: Callable[[np.array], np.array] = lambda data: data[:,2]
        petal_width: Callable[[np.array], np.array] = lambda data: data[:,3]

        field_data: Dict[str, pd.DataFrame] = {
            "sepal_width": pd.DataFrame.from_dict({k:sepal_width(v) for (k, v) in raw_data.items()}),
            "sepal_length": pd.DataFrame.from_dict({k:sepal_length(v) for (k, v) in raw_data.items()}),
            "petal_width": pd.DataFrame.from_dict({k:petal_width(v) for (k, v) in raw_data.items()}),
            "petal_length": pd.DataFrame.from_dict({k:petal_length(v) for (k, v) in raw_data.items()})
        }
        return field_data

    @staticmethod
    def compare_species_by_feature_pair(
        raw_data: Dict[str, pd.DataFrame], 
        f1: str, 
        f2: str, 
        ttl: str,
        color_dict: Dict[str, str]=dict(setosa="#e41a1c", versicolor="#377eb8", virginica="#4daf4a")) -> None:
        field_data: Dict[str, pd.DataFrame] = IrisOps.arrange_by_field(raw_data)

        def load_traces(fig: go.Figure) -> go.Figure:
            field1: pd.DataFrame = field_data[f1]
            field2: pd.DataFrame = field_data[f2]
            for col in field1.columns:
                fig.add_trace(
                    go.Scatter(
                        x=field1[col],
                        y=field2[col],
                        name=col,
                        mode="markers",
                        marker_color=color_dict[col]
                    )
                )
            return fig

        fig: go.Figure = load_traces(go.Figure())
        fig.update_layout(title=ttl, template="plotly_white", xaxis_title_text=f1, yaxis_title_text=f2)
        fig.show()

    @staticmethod
    def value_overlap_matrix(data: pd.DataFrame) -> np.array:
        """
        Takes a DataFrame of presumably one data type, and determines the extent to
        which each column overlaps with each other. For example, the first element
        of the matrix (i.e. data[0][0]) would be 1, since a series overlaps with 
        itself entirely. If half of the data in the second column is greater than
        or equal to the first column's minimum and less than or equal to the first
        column's maximum, the second element (i.e. data[0][1]) would be 0.5. By this
        scheme, all of the proportions for the ith *column* of the input data reside in 
        the ith *row* of the output matrix.
        """
        ncols: int = len(data.columns)
        out: np.array = np.zeros((ncols, ncols))
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                col1_max: Rational = Vector(data[col1]).max()
                col1_min: Rational = Vector(data[col1]).min()
                col2_overlap: Callable[[Rational], bool] = lambda x: (x <= col1_max) & (x >= col1_min)
                col2_prop: Rational = Vector(data[col2]).conditional_prop(lambda x: col2_overlap(x)) 
                out[i][j] = col2_prop
        return out