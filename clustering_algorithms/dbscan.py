import sys
import numpy as np
sys.path.append('../data_loader')
sys.path.append('../data_visualization')

from clustering_algorithm import ClusteringAlgorithm
from data_loader import DataLoader
from visualizer import Visualizer


class DBSCAN(ClusteringAlgorithm):
    def __init__(self, coordinates, labels, epsilon=0.5, min_points=5) -> None:
        super().__init__()
        self.__coordinates = coordinates
        self.__labels = labels
        self.__epsilon = epsilon
        self.__min_points = min_points
        self.__predicted_labels = np.full(len(self.__labels), -2)

    def __compute_euclidean_dist(self, point_a, point_b):
        diff = point_a - point_b
        sum_of_squares = np.dot(diff.T, diff)
        euclidean_dist = np.sqrt(sum_of_squares)

        return euclidean_dist

    def __get_neighbours(self, current_index):
        neighbours = np.array([], dtype=int)
        point = self.__coordinates[current_index]

        for i in range(self.__labels.size):
            if i == current_index:
                continue

            dist = self.__compute_euclidean_dist(point, self.__coordinates[i])

            if dist < self.__epsilon:
                neighbours = np.append(neighbours, i)

        return neighbours

    def __expand_cluster(self, initial_point_index, neighbours, cluster_index):
        for neighbour in neighbours[initial_point_index]:
            if self.__predicted_labels[neighbour] == -1:
                self.__predicted_labels[neighbour] = cluster_index
                continue 

            if self.__predicted_labels[neighbour] != -2:
                continue 
            
            neighbours[neighbour] = self.__get_neighbours(neighbour)
            self.__predicted_labels[neighbour] = cluster_index

            if neighbours[neighbour].size < self.__min_points:
                continue

            else:
                self.__expand_cluster(neighbour, neighbours, cluster_index)


    def fit_transform(self):
        cluster_index = -1
        neighbours = {}

        for index in range(self.__predicted_labels.size):
            if self.__predicted_labels[index] != -2:
                continue

            neighbours[index] = self.__get_neighbours(index)

            if neighbours[index].size < self.__min_points:
                self.__predicted_labels[index] = -1
                continue

            cluster_index += 1
            self.__predicted_labels[index] = cluster_index
            self.__expand_cluster(index, neighbours, cluster_index)

    def score(self, scorer):
        return super().score(scorer)

    @property
    def predicted_labels(self):
        return self.__predicted_labels


def main():
    # load data
    data_loader = DataLoader('smile')

    # fit DBSCAN
    dbscan = DBSCAN(*data_loader.get_data())
    dbscan.fit_transform()

    # visualize initial 2D data and predictions
    Visualizer.plot_dataset_2d(data_loader.data_2d, data_loader.dataset_name)
    Visualizer.plot_predictions_dataset_2d(data_loader.data_2d, data_loader.dataset_name, dbscan.predicted_labels)

if __name__ == '__main__':
    main()
