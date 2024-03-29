from dbscan import DBSCAN
from kmeans import KMeans
from agglomerative import Agglomerative
from gaussian_mixture import GaussianMixture
from visualizer import Visualizer
from data_loader import DataLoader
from data_viz_preparation import DataVizPreparation
from scorer import Scorer


class Clustering:
    def __init__(self, dataset_name, algorithm_name, algorithm=None) -> None:
        self.__dataset_name = dataset_name
        self.__data_loader = DataLoader(self.__dataset_name)
        self.__algorithm_name = algorithm_name

        if algorithm is not None:
            self.__algorithm = algorithm
        else:
            self.__algorithm = self.__get_algorithm()

    def __get_algorithm(self):
        algorithm = None

        if self.__algorithm_name == "kmeans":
            algorithm = KMeans(*self.__data_loader.data)
        elif self.__algorithm_name == "dbscan":
            algorithm = DBSCAN(*self.__data_loader.data)
        elif self.__algorithm_name == "agglomerative":
            algorithm = Agglomerative(*self.__data_loader.data)
        elif self.__algorithm_name == "gaussian mixture":
            algorithm = GaussianMixture(*self.__data_loader.data)
        else:
            raise NameError("clustering algorithm not found")

        return algorithm

    def train(self):
        print(f"Training {self.__algorithm_name} clustering...")
        self.__algorithm.fit_transform()

    def visualize_result(self):
        agglomerative = "agglomerative" == self.__algorithm_name

        clusters_evolution = DataVizPreparation.prepare_viz(
            self.__algorithm, self.__data_loader.data_2d
        )
        Visualizer.plot_custering_evolution(
            clusters_evolution, f"{self.__algorithm_name} evolution", agglomerative
        )

        if self.__algorithm_name == "agglomerative":
            Visualizer.plot_dendrogram(
                self.__algorithm.linkage_matrix, self.__dataset_name
            )

    def score(self):
        scorer = Scorer(
            self.__data_loader.data[0],
            self.__algorithm.predicted_labels,
            self.__data_loader.data[1],
        )
        scorer.within_cluster_variation()
        scorer.rand_index()
        scorer.purity()
