from re import T
import numpy as np
from scipy.stats import multivariate_normal
from clustering_algorithm import ClusteringAlgorithm


class GaussianMixture(ClusteringAlgorithm):
    def __init__(self, coordinates, labels, n_clusters=3, max_iter=500) -> None:
        super().__init__()
        self.__coordinates = np.array(coordinates)
        self.__n_clusters = n_clusters
        self.__max_iters = max_iter
        self.__cluster_weights = None
        self.__points_weights = None
        self.__mu = None
        self.__sigma = None
        self.__predicted_labels = None
        self.__labels_evolution = []

    def __initialize_gaussians(self):
        init_value = 1 / self.__n_clusters
        self.__cluster_weights = np.full(self.__n_clusters, init_value)
        self.__points_weights = np.full(self.__coordinates.shape, init_value)

        indices = np.random.randint(low=0, high=len(
            self.__coordinates), size=self.__n_clusters)
        self.__mu = [self.__coordinates[i] for i in indices]
        self.__sigma = [np.cov(self.__coordinates.T)
                        for _ in range(self.__n_clusters)]

    def __expectation(self):
        likelihood = np.zeros((len(self.__coordinates), self.__n_clusters))
        for i in range(self.__n_clusters):
            gaussian = multivariate_normal(self.__mu[i], self.__sigma[i])
            likelihood[:, i] = gaussian.pdf(self.__coordinates)

        numerator = likelihood * self.__cluster_weights
        denominator = numerator.sum(axis=1)[:, np.newaxis]

        self.__points_weights = numerator / denominator
        labels = [np.argmax(prob) for prob in self.__points_weights]
        self.__labels_evolution.append(labels)
        self.__cluster_weights = self.__points_weights.mean(axis=0)

    def __maximization(self):
        for i in range(self.__n_clusters):
            weights = self.__points_weights[:, [i]]
            weigths_sum = weights.sum()

            self.__mu[i] = (self.__coordinates * weights).sum(axis=0)
            self.__mu[i] /= weigths_sum

            self.__sigma[i] = np.cov(self.__coordinates.T, aweights=(
                weights/weigths_sum).flatten(), bias=True)

    def fit_transform(self):
        self.__initialize_gaussians()

        for _ in range(self.__max_iters):
            self.__expectation()
            self.__maximization()

            if len(self.__labels_evolution) >= 2:
                if self.__labels_evolution[-1] == self.__labels_evolution[-2]:
                    break

        self.__predicted_labels = self.__labels_evolution[-1]

    def score(self, scorer):
        return super().score(scorer)

    @property
    def predicted_labels(self):
        return self.__predicted_labels

    @property
    def labels_evolution(self):
        return self.__labels_evolution
