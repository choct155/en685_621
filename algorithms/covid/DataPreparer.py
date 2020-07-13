import pandas as pd
from typing import Sequence

class DataPreparer:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data

    def convert_to_long(self, drop_cols: Sequence[str], idx_cols: Sequence[str]) -> pd.DataFrame:
        long_data = pd.DataFrame(
        self.data.drop(drop_cols, axis=1)
            .set_index(idx_cols)
            .stack()
        ).reset_index()
        long_data.columns = [s.lower() for s in idx_cols] + ["date", "count"]
        us_data: pd.DataFrame = long_data[~long_data["admin2"].isnull()]
        us_data.columns = ["city", "state", "country", "date", "count"]
        return us_data

    def __series_by_city(self, long_data: pd.DataFrame, state: str) -> pd.DataFrame:
        agg_by_city: pd.DataFrame = (
            long_data
                .groupby(["city", "state", "date"])
                .sum()
                .reset_index()
        )
        agg_by_city.columns = ["city", "state", "date", "count"]
        agg_by_city["date"] = pd.to_datetime(agg_by_city["date"])
        city_series: pd.DataFrame = (
            agg_by_city[agg_by_city["state"] == state]
                .set_index(["date", "city"])

        )
        out: pd.DataFrame = city_series["count"].unstack("city")
        return out

    def __series_by_state(self, long_data: pd.DataFrame) -> pd.DataFrame:
        agg_by_state: pd.DataFrame = (
            long_data
            .groupby(["state", "date"])
            .sum()
            .reset_index()
        )
        agg_by_state.columns = ["state", "date", "count"]
        agg_by_state["date"] = pd.to_datetime(agg_by_state["date"])
        state_series: pd.DataFrame = agg_by_state.set_index(["date", "state"]).unstack("state")
        state_series.columns = [col[1] for col in state_series.columns]
        return state_series

    def to_city_series(self, drop_cols: Sequence[str], idx_cols: Sequence[str], state: str) -> pd.DataFrame:
        long_data: pd.DataFrame = self.convert_to_long(drop_cols, idx_cols)
        return self.__series_by_city(long_data, state)

    def to_state_series(self, drop_cols: Sequence[str], idx_cols: Sequence[str]) -> pd.DataFrame:
        long_data: pd.DataFrame = self.convert_to_long(drop_cols, idx_cols)
        return self.__series_by_state(long_data)

    def process_recovered(self) -> pd.DataFrame:
        us_data: pd.DataFrame = (
            self.data[self.data["Country/Region"] == "US"]
                .set_index(["Province/State", "Country/Region", "Lat", "Long"])
                .stack()
                .reset_index()
        )
        us_data.columns = ["state", "country", "lat", "long", "date", "count"]
        us_data["date"] = pd.to_datetime(us_data["date"])
        return us_data[["date", "count"]].set_index("date")["count"]