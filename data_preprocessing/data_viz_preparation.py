import pandas as pd
from copy import deepcopy


class DataVizPreparation:
    @staticmethod
    def prepare_kmeans_viz(kmeans, data_2d):
        data = data_2d.iloc[:, :-1]
        points_no = data.shape[0]
        iter_no = len(kmeans.labels_evolution)
        iteration_index = [[i] * points_no for i in range(iter_no)]

        clusters_evolution = None

        for i in range(iter_no):
            current_data = deepcopy(data)
            current_data['class'] = kmeans.labels_evolution[i]
            current_data['iteration'] = iteration_index[i]

            if clusters_evolution is None:
                clusters_evolution = current_data
            else:
                clusters_evolution = pd.concat(
                    [clusters_evolution, current_data])

        return clusters_evolution
