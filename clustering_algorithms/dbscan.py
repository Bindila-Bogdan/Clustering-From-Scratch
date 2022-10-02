from dis import dis
import numpy as np

from clustering_algorithm import ClusteringAlgorithm
from data_loader import DataLoader
from visualizer import Visualizer
from data_viz_preparation import DataVizPreparation


class DBSCAN(ClusteringAlgorithm):
    def __init__(self, coordinates, labels, epsilon=0.5, min_points=3, distance_type='euclidean') -> None:
        super().__init__()
        self.__coordinates = coordinates
        self.__labels = labels
        self.__epsilon = epsilon
        self.__min_points = min_points
        self.__distance_type = distance_type
        self.__predicted_labels = np.full(len(self.__labels), -2)
        self.__labels_evolution = []

    def __get_neighbours(self, current_index):
        neighbours = np.array([], dtype=int)
        point = self.__coordinates[current_index]

        for i in range(self.__labels.shape[0]):
            if i == current_index:
                continue

            dist = super().compute_dist(point, self.__coordinates[i], self.__distance_type)

            if dist < self.__epsilon:
                neighbours = np.append(neighbours, i)

        return neighbours

    def __expand_cluster(self, initial_point_index, neighbours, cluster_index):
        for neighbour in neighbours[initial_point_index]:
            if self.__predicted_labels[neighbour] == -1:
                self.__predicted_labels[neighbour] = cluster_index
                self.__labels_evolution.append(list(self.__predicted_labels))
                continue

            if self.__predicted_labels[neighbour] != -2:
                continue

            neighbours[neighbour] = self.__get_neighbours(neighbour)
            self.__predicted_labels[neighbour] = cluster_index
            self.__labels_evolution.append(list(self.__predicted_labels))

            if neighbours[neighbour].shape[0] < self.__min_points:
                continue

            else:
                self.__expand_cluster(neighbour, neighbours, cluster_index)

    def fit_transform(self):
        cluster_index = -1
        neighbours = {}

        for index in range(self.__predicted_labels.shape[0]):
            if self.__predicted_labels[index] != -2:
                continue

            neighbours[index] = self.__get_neighbours(index)

            if neighbours[index].shape[0] < self.__min_points:
                self.__predicted_labels[index] = -1
                self.__labels_evolution.append(list(self.__predicted_labels))
                continue

            cluster_index += 1
            self.__predicted_labels[index] = cluster_index
            self.__labels_evolution.append(list(self.__predicted_labels))
            self.__expand_cluster(index, neighbours, cluster_index)

    def score(self, scorer):
        return super().score(scorer)

    @property
    def predicted_labels(self):
        return self.__predicted_labels

    @property
    def labels_evolution(self):
        return self.__labels_evolution

    def __str__(self) -> str:
        description = '*DBSCAN*\n'
        description += f'epsilon: {self.__epsilon}\n'
        description += f'min_points: {self.__min_points}\n'
        description += f'distance_type: {self.__distance_type}'

        return description