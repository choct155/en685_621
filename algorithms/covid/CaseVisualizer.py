import pandas as pd
import plotly.graph_objects as go

class CaseVisualizer:

    def __init__(
        self, 
        measure: str, 
        confirmed: pd.DataFrame, 
        deaths: pd.DataFrame,
        recovered: pd.DataFrame,
        index: pd.DatetimeIndex
        ) -> None:
        self.measure: str = measure
        self.confirmed: pd.DataFrame = confirmed
        self.deaths: pd.DataFrame = deaths
        self.recovered: pd.DataFrame = recovered
        self.index: pd.DatetimeIndex = index

    def load(self) -> None:
        self.data: pd.DataFrame = pd.DataFrame({
            "confirmed": self.confirmed[self.measure],
            "deaths": self.deaths[self.measure],
            "recovered": self.recovered[self.measure],
            "date": self.index
        })

    def plot_measure_comparison(self, ttl: str) -> None:
        fig = go.Figure()
        for status in ["confirmed", "deaths", "recovered"]:
            fig.add_trace(
                go.Scatter(
                    x=self.data["date"],
                    y=self.data[status],
                    mode="lines",
                    name=status
                )
            )
        fig.update_layout(
            title=ttl,
            template='plotly_white'
        )
        fig.show()