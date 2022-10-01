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

    def __compute_euclidean_dist(self, point_a, point_b):
        diff = point_a - point_b
        sum_of_squares = np.dot(diff.T, diff)
        euclidean_dist = np.sqrt(sum_of_squares)

        return euclidean_dist

    def __compute_manhattan_dist(self, point_a, point_b):
        diff = np.abs(point_a - point_b)
        manhattan_dist = diff.sum()

        return manhattan_dist

    def __compute_cosine_similarity_dist(self, point_a, point_b):
        numerator = np.dot(point_a, point_b)
        denominator = np.linalg.norm(point_a) * np.linalg.norm(point_b)
        cosine_similarity = numerator / denominator
        cosine_similarity_dist = 1 - cosine_similarity

        return cosine_similarity_dist

    def __compute_pearson_corr_dist(self, point_a, point_b):
        corr_matrix = np.corrcoef(point_a, point_b)
        pearson_corr = corr_matrix[0][1]
        pearson_corr_dist = 1 - pearson_corr

        return pearson_corr_dist

    def __compute_dist(self, point_a, point_b, distance_type='euclidean'):
        if distance_type == 'manhattan':
            return self.__compute_manhattan_dist(point_a, point_b)
        elif distance_type == 'cosine_similarity':
            return self.__compute_cosine_similarity_dist(point_a, point_b)
        elif distance_type == 'pearson_correlation':
            return self.__compute_pearson_corr_dist(point_a, point_b)
        else:
            return self.__compute_euclidean_dist(point_a, point_b)
