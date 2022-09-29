import numpy as np
from scipy.stats import multivariate_normal
from clustering_algorithm import ClusteringAlgorithm


class GaussianMixtureModel(ClusteringAlgorithm):
    def __init__(self, coordinates, labels, n_clusters=3, max_iter=100) -> None:
        super().__init__()
        self.__coordinates = np.array(coordinates)
        self.__labels = labels
        self.__n_clusters = n_clusters
        self.__max_iters = max_iter
        self.__cluster_weights = None
        self.__points_weights = None
        self.__mu = None
        self.__sigma = None

    def __initialize_gaussians(self):
        init_value = 1 / self.__n_clusters
        # why do we have cluster weights?
        self.__cluster_weights = np.full(self.__n_clusters, init_value)
        self.__points_weights = np.full(self.__coordinates.shape, init_value)

        indices = np.random.randint(low=0, high=len(self.__coordinates), size=self.__n_clusters)
        self.__mu = [self.__coordinates[i] for i in indices]
        self.__sigma = [np.cov(self.__coordinates.T) for _ in range(self.__n_clusters)]

    def __expectation(self):
        likelihood = np.zeros((len(self.__coordinates), self.__n_clusters))
        for i in range(self.__n_clusters):
            gaussian =  multivariate_normal(self.__mu[i], self.__sigma[i])
            likelihood[:, i] = gaussian.pdf(self.__coordinates)

        numerator = likelihood * self.__cluster_weights
        denominator = numerator.sum(axis=1)[:, np.newaxis]

        self.__points_weights = numerator / denominator
        self.__cluster_weights = self.__points_weights.mean(axis=0)
        

    def __maximization(self):
        pass

    def fit_transform(self):
        self.__initialize_gaussians()

        for i in range(self.__max_iters):
            self.__expectation()
            self.__maximization()

            break

    def score(self, scorer):
        return super().score(scorer)
