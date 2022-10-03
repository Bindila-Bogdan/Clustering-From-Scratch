import sys
import numpy as np
from gaussian_mixture import GaussianMixture
from dbscan import DBSCAN
from kmeans import KMeans
from scorer import Scorer


class Optimizer:
    def __init__(self, algorithm_name, coordinates, labels, optimization_metric) -> None:
        self.__algorithm_name = algorithm_name
        self.__coordinates = coordinates
        self.__labels = labels
        self.__metric = optimization_metric

    def __optimize_kmeans(self, best_score):
        best_kmeans = None
        clusters_no = len(set(self.__labels))
        distance_metrics = ['euclidean', 'manhattan',
                            'cosine_similarity', 'pearson_correlation']

        for distance_metric in distance_metrics:
            print(f'distance metric: {distance_metric} -> ', end='')
            best_kmeans_instance = None

            if self.__metric == 'wss':
                max_score_instance = sys.maxsize
            else:
                max_score_instance = -sys.maxsize

            for _ in range(3):
                kmeans = KMeans(self.__coordinates, self.__labels,
                                n_clusters=clusters_no, distance_type=distance_metric)
                kmeans.fit_transform()
                scorer = Scorer(self.__coordinates, kmeans.predicted_labels,
                                labels=self.__labels, display=False)
                score = scorer.get_score(self.__metric)

                if (score < max_score_instance and self.__metric == 'wss') or \
                        (score > max_score_instance and self.__metric != 'wss'):
                    max_score_instance = score
                    best_kmeans_instance = kmeans
                else:
                    del kmeans

                del scorer

            print(round(max_score_instance, 4))

            if (max_score_instance < best_score and self.__metric == 'wss') or \
                    (max_score_instance > best_score and self.__metric != 'wss'):
                best_score = max_score_instance
                best_kmeans = best_kmeans_instance
            else:
                del best_kmeans_instance

        print(best_kmeans)

        return best_kmeans, best_score

    def __optimize_dbscan(self, best_score):
        best_dbscan = None
        epsilons = [0.25, 0.5, 0.75]
        min_points = [3, 5, 7]
        distance_metrics = ['euclidean', 'manhattan',
                            'cosine_similarity', 'pearson_correlation']

        for eps in epsilons:
            for min_point in min_points:
                for distance_metric in distance_metrics:
                    print(f'eps: {eps} | min_points: {min_point} | distance metric: {distance_metric} -> ', end='')
                    dbscan = DBSCAN(self.__coordinates, self.__labels, epsilon=eps,
                                    min_points=min_point, distance_type=distance_metric)
                    dbscan.fit_transform()

                    scorer = Scorer(self.__coordinates, dbscan.predicted_labels,
                                    labels=self.__labels, display=False)
                    score = scorer.get_score(self.__metric)

                    if (score < best_score and self.__metric == 'wss') or \
                            (score > best_score and self.__metric != 'wss'):
                        best_score = score
                        best_dbscan = dbscan
                    else:
                        del dbscan

                    del scorer

                    print(round(score, 4))

        print(best_dbscan)

        return best_dbscan, best_score

    def __optimize_gaussian_mixture(self, best_score):
        best_gaussian_mixture = None
        clusters_no = len(set(self.__labels))

        for seed in range(1, 4):
            print(f'seed: {seed} -> ', end='')
            try:
                gaussian_mixture = GaussianMixture(self.__coordinates, self.__labels, n_clusters=clusters_no, seed=seed)
                gaussian_mixture.fit_transform()

                scorer = Scorer(self.__coordinates, gaussian_mixture.predicted_labels,
                                labels=self.__labels, display=False)
                score = scorer.get_score(self.__metric)

                if (score < best_score and self.__metric == 'wss') or \
                        (score > best_score and self.__metric != 'wss'):
                    best_score = score
                    best_gaussian_mixture = gaussian_mixture
                else:
                    del gaussian_mixture

                del scorer

            except np.linalg.LinAlgError:
                print('error')
                continue

            print(round(score, 4))

        print(best_gaussian_mixture)

        return best_gaussian_mixture, best_score

    def optimize(self):
        print(f'Optimizing {self.__algorithm_name} based on {self.__metric}')

        if self.__metric == 'wss':
            best_score = sys.maxsize
        else:
            best_score = -sys.maxsize

        if self.__algorithm_name == 'kmeans':
            return self.__optimize_kmeans(best_score)
        elif self.__algorithm_name == 'dbscan':
            return self.__optimize_dbscan(best_score)
        elif self.__algorithm_name == 'gaussian mixture':
            return self.__optimize_gaussian_mixture(best_score)
