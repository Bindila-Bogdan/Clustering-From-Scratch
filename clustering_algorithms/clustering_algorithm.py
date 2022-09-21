import numpy as np
import sys
from abc import ABC, abstractmethod
sys.path.append('../data_preprocessing')
sys.path.append('../data_visualization')


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def fit_transform(self):
        pass

    # TODO: impute the scoring method
    @abstractmethod
    def score(self, scorer):
        pass

    def compute_euclidean_dist(self, point_a, point_b):
        diff = point_a - point_b
        sum_of_squares = np.dot(diff.T, diff)
        euclidean_dist = np.sqrt(sum_of_squares)

        return euclidean_dist
