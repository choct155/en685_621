import pandas as pd
from typing import Sequence, Dict
from algorithms.covid.Config import Config

class CovidReader:

    def __init__(self, config: Config) -> None:
        self.file_paths: Dict[str, str] = config.data_files

    def load(self) -> Dict[str, pd.DataFrame]:
        return {k:pd.read_csv(v) for (k, v) in self.file_paths.items()} 