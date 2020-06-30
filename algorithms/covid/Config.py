from dataclasses import dataclass
from typing import Dict

class Config:
    """Holds potentially variable input information for the COVID-19 analysis"""

    def __init__(self):
        self.project_dir: str = "/home/choct155/projects/math/algorithms/algoforDS/en685_621/"
        self.data_dir: str = f"{self.project_dir}data/"
        self.data_files: Dict[str, str] = dict(
            confirmed = f"{self.data_dir}time_series_covid_19_confirmed_US.csv",
            deaths = f"{self.data_dir}time_series_covid_19_deaths_US.csv",
            recovered = f"{self.data_dir}datasets_494724_1273275_time_series_covid_19_recovered.csv"
        )   