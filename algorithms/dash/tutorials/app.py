import dash
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go 
import numpy as np
import pandas as pd
from algorithms.iris.IrisOps import IrisOps
from typing import List

external_stylesheets: List[str] = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app: Dash = Dash(__name__, external_stylesheets=external_stylesheets)

df: pd.DataFrame = pd.DataFrame({
    "Fruit": ["apples", "oranges", "bananas", "apples", "oranges", "bananas"],
    "Amount": [4,1,2,2,4,5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fruits: np.array = np.array(["Apples", "Oranges", "Bananas"])
amounts: np.array = np.array([
    [2,4,2],
    [1,5,3]
])
cities: np.array = np.array(["San Francisco", "Montreal"])

fig: go.Figure = go.Figure()
for i, city in enumerate(cities):
    fig.add_trace(go.Bar(
        x = fruits,
        y = amounts[i],
        name=city
    ))
fig.update_layout(
    barmode="group",
    template="plotly_white"
)

app.layout = html.Div(
    children = [
        html.H1(
            children="Fisher Price",
            style={
                "textAlign": "center"
            }
        ),
        html.Div(
            children="""My First Dash Application""",
            style={
                "textAlign": "center"
            }
        ),
        dcc.Graph(
            id="example-graph",
            figure=fig
        )
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
