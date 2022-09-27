from fileinput import close
import sys
import numpy as np
from clustering_algorithm import ClusteringAlgorithm


class Agglomerative(ClusteringAlgorithm):
    def __init__(self, coordinates, labels) -> None:
        super().__init__()
        self.__coordinates = coordinates
        self.__labels = labels
        self.__predicted_labels = list(range(len(self.__labels)))
        self.__labels_evolution = []
        self.__clusters = [[i] for i in list(range(len(self.__labels)))]
        self.__centroids = []


    def __compute_centroids(self):
        self.__centroids = []

        for cluster in self.__clusters:
            centroid = np.array([self.__coordinates[i] for i in cluster]).mean(axis=0)
            print(centroid)
            self.__centroids.append(centroid)


    def __get_closest_clusters(self):
        closest_clusters = [None, None]
        min_dist = sys.maxsize

        for i in range(len(self.__clusters)):
            for j in range(len(self.__clusters)):
                if i == j:
                    continue
            
                centorid_i = self.__centroids[i]
                centroid_j = self.__centroids[j]
                dist = super().compute_euclidean_dist(centorid_i, centroid_j)
                if dist < min_dist:
                    min_dist = dist
                    closest_clusters = [i, j]

        return closest_clusters

    def fit_transform(self):
        self.__compute_centroids()
        print(self.__centroids)
        closest_clusters = self.__get_closest_clusters()

    def score(self, scorer):
        return super().score(scorer)
