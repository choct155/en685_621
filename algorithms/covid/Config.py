from dataclasses import dataclass
from typing import Dict

@dataclass 
class Config:
    """Holds potentially variable input information for the COVID-19 analysis"""
    project_dir: str = "/home/choct155/projects/math/algorithms/algoforDS/en685_621/"
    data_dir: str = f"{project_dir}data/"
    data_files: Dict[str, str] = dict(
        confirmed = f"{data_dir}time_series_covid_19_confirmed_US.csv",
        deaths = f"{data_dir}time_series_covid_19_deaths_US.csv",
        recovered = f"{data_dir}datasets_494724_1273275_time_series_covid_19_recovered.csv"
    )   