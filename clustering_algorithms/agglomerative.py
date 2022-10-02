import sys
from copy import deepcopy
import numpy as np
from clustering_algorithm import ClusteringAlgorithm


class Agglomerative(ClusteringAlgorithm):
    def __init__(self, coordinates, labels, distance_type='euclidean') -> None:
        super().__init__()
        self.__coordinates = coordinates
        self.__labels = labels
        self.__distance_type = distance_type
        self.__clusters = [[i] for i in list(range(len(self.__labels)))]
        self.__linkage_matrix = []
        self.__linkage_matrix_indices = list(range(len(self.__labels)))
        self.__centroids = []
        self.__labels_evolution = [list(range(len(self.__coordinates)))]

    def __compute_centroids(self):
        self.__centroids = []

        for cluster in self.__clusters:
            centroid = np.array([self.__coordinates[i]
                                for i in cluster]).mean(axis=0)
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
                dist = super().compute_dist(centorid_i, centroid_j, self.__distance_type)
                if dist < min_dist:
                    min_dist = dist
                    closest_clusters = [i, j]

        return closest_clusters, min_dist

    def __merge_clusters(self, closest_clusters, min_dist):
        i = closest_clusters[0]
        j = closest_clusters[1]

        linkage_i = self.__linkage_matrix_indices[i]
        linkage_j = self.__linkage_matrix_indices[j]
        new_cluster_size = len(self.__clusters[i]) + len(self.__clusters[j])
        linkage_entry = [linkage_i, linkage_j, min_dist, new_cluster_size]

        self.__linkage_matrix.append(linkage_entry)

        labels = deepcopy(self.__labels_evolution[-1])
        color_cluster_i = labels[self.__clusters[i][0]]

        for index in self.__clusters[j]:
            labels[index] = color_cluster_i

        self.__labels_evolution.append(labels)

        self.__clusters[i].extend(self.__clusters[j])
        self.__clusters.pop(j)

        max_linkage_index = max(self.__linkage_matrix_indices) + 1
        self.__linkage_matrix_indices[i] = max_linkage_index
        self.__linkage_matrix_indices.pop(j)

    def fit_transform(self):
        while len(self.__clusters) != 1:
            self.__compute_centroids()
            closest_clusters, min_dist = self.__get_closest_clusters()
            self.__merge_clusters(closest_clusters, min_dist)

    def score(self, scorer):
        return super().score(scorer)

    @property
    def linkage_matrix(self):
        return np.array(self.__linkage_matrix)

    @property
    def labels_evolution(self):
        return self.__labels_evolution

    def __str__(self) -> str:
        description = '*Agglomerative*\n'
        description += f'distance_type: {self.__distance_type}'

        return description