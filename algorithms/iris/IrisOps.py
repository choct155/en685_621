import numpy as np
from typing import Dict, Tuple
from algorithms.iris.DataGenerator import DataGenerator
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
        # ax.scatter(IrisOps.getCol(input_data, features[0]), IrisOps.getCol(input_data, features[1]), c='r')
        # ax.scatter(IrisOps.getCol(synth_data, features[0]), IrisOps.getCol(synth_data, features[1]), c='b')