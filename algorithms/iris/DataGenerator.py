import numpy as np
from algorithms.iris.AttrStats import Stats
from algorithms.iris.AttrStats import AttrStats

class DataGenerator:

    @staticmethod
    def align_mean(data: np.array, target_mean: np.array) -> np.array:
        input_mean: np.array = np.mean(data, axis=0)
        mean_diff: np.array = input_mean - target_mean
        return data - mean_diff

    @staticmethod
    def min_max_norm(data: np.array, min_vals: np.array, max_vals: np.array) -> np.array:
        input_min: np.array = np.min(data, axis=0)
        input_max: np.array = np.max(data, axis=0)
        data_in_unit_itvl: np.array = (data - input_min) / (input_max - input_min)
        target_range: np.array = max_vals - min_vals
        return (data_in_unit_itvl * target_range) + input_min

    @staticmethod
    def normalize(data: np.array, stats: Stats) -> np.array:
        with_target_mean: np.array = DataGenerator.align_mean(data, stats.mean)
        with_target_range: np.array = DataGenerator.min_max_norm(with_target_mean, stats.min, stats.max)
        return with_target_range

    @staticmethod
    def gen_norm_data(stats: Stats, n: int = 100) -> np.array:
        ncol: int = stats.mean.size
        rand_data: np.array = np.random.uniform(size = ncol*n).reshape(n, ncol)
        norm_data: np.array = DataGenerator.normalize(rand_data, stats)
        return norm_data

    @staticmethod
    def align_covariance(data: np.array, stats: Stats) -> np.array:
        return data.dot(stats.cov)

    @staticmethod
    def gen_synthetic_data(input_data: np.array, n: int = 100) -> np.array:
        attr_stats: AttrStats = AttrStats(input_data)
        input_stats: Stats = attr_stats.describe()
        init_synthetic: np.array = DataGenerator.gen_norm_data(input_stats, n)
        cov_adjusted: np.array = DataGenerator.align_covariance(init_synthetic, input_stats)
        second_norm: np.array = DataGenerator.normalize(cov_adjusted, input_stats)
        return second_norm
        