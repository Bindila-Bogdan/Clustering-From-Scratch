import sys
import numpy as np
from copy import deepcopy


from clustering_algorithm import ClusteringAlgorithm
from data_loader import DataLoader
from visualizer import Visualizer
from data_viz_preparation import DataVizPreparation


class KMeans(ClusteringAlgorithm):
    def __init__(self, coordinates, labels, n_clusters=3, max_iter=100) -> None:
        super().__init__()
        self.__coordinates = coordinates
        self.__labels = labels
        self.__n_clusters = n_clusters
        self.__max_iters = max_iter
        self.__predicted_labels = np.full(len(self.__labels), -1)
        self.__labels_evolution = []

    def __kmeans_initialization(self):
        coordinates = deepcopy(self.__coordinates)
        first_index = np.random.choice(coordinates.shape[0])

        centroids = [coordinates[first_index, :]]
        coordinates = np.delete(coordinates, first_index, axis=0)

        while len(centroids) < self.__n_clusters:
            dist_distrib = []

            for i in range(coordinates.shape[0]):
                min_dist = sys.maxsize

                for centroid in centroids:
                    min_dist = min(
                        min_dist, super().compute_euclidean_dist(coordinates[i, :], centroid))

                dist_distrib.append(min_dist)

            distrib_sum = sum(dist_distrib)
            dist_distrib /= distrib_sum

            centroid_index = np.random.choice(
                coordinates.shape[0], p=dist_distrib)
            centroids.append(coordinates[centroid_index, :])
            coordinates = np.delete(coordinates, centroid_index, axis=0)

        return centroids

    def fit_transform(self):
        prev_centroids = self.__kmeans_initialization()
        iter_no = 0

        while iter_no < self.__max_iters:
            points_class = []

            for point_index in range(self.__coordinates.shape[0]):
                min_dist = sys.maxsize
                closest_centroid_index = None

                for centorid_index, centroid in enumerate(prev_centroids):
                    dist = super().compute_euclidean_dist(
                        self.__coordinates[point_index, :], centroid)
                    if dist < min_dist:
                        min_dist = dist
                        closest_centroid_index = centorid_index

                points_class.append(closest_centroid_index)

            new_centroids = []

            for class_value in set(points_class):
                points_in_class = []

                for index in range(len(points_class)):
                    if points_class[index] == class_value:
                        points_in_class.append(self.__coordinates[index])

                new_centroids.append(
                    np.mean(np.array(points_in_class), axis=0))

            equalities = [True if all(
                prev_centroids[i] == new_centroids[i]) else False for i in range(len(new_centroids))]
            if sum(equalities) == len(new_centroids):
                break
            prev_centroids = new_centroids

            self.__labels_evolution.append(points_class)
            iter_no += 1

        self.__predicted_labels = points_class

    def score(self, scorer):
        return super().score(scorer)

    @property
    def predicted_labels(self):
        return self.__predicted_labels

    @property
    def labels_evolution(self):
        return self.__labels_evolution


def main():
    # load data
    data_loader = DataLoader('iris')

    # fit KMeans
    kmeans = KMeans(*data_loader.data)
    kmeans.fit_transform()

    # visualize KMeans evolution
    clusters_evolution = DataVizPreparation.prepare_kmeans_viz(
        kmeans, data_loader.data_2d)
    Visualizer.plot_kmeans_evolution(clusters_evolution)


if __name__ == '__main__':
    main()
