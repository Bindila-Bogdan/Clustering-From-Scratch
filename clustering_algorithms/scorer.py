import sys
import numpy as np


class Scorer:
    def __init__(
        self, coordinates, predicted_labels, labels=None, display=True
    ) -> None:
        self.__coordinates = coordinates
        self.__predicted_labels = predicted_labels
        self.__labels = labels
        self.__clusters_labels = self.__get_cluster_labels()
        self.__display = display

    def __get_cluster_labels(self):
        clusters_labels = sorted(list(set(self.__predicted_labels)))

        try:
            clusters_labels.remove(-1)
        except ValueError:
            pass

        return clusters_labels

    def within_cluster_variation(self):
        wss = 0
        centroids = []

        if len(set(self.__clusters_labels)) == 0:
            if self.__display:
                print("0 clusters")

            return sys.maxsize

        for cluster_label in self.__clusters_labels:
            points = [
                self.__coordinates[i]
                for i, value in enumerate(self.__predicted_labels)
                if value == cluster_label
            ]

            centroid = np.array(points).mean(axis=0)
            centroids.append(centroid)

            for point in points:
                diff = centroid - point
                euclidean_dist = np.sqrt(np.dot(diff.T, diff))

                wss += euclidean_dist

        for i in range(len(self.__coordinates)):
            if self.__predicted_labels[i] == -1:
                min_dist = sys.maxsize

                for centroid in centroids:
                    diff = centroid - self.__coordinates[i]
                    euclidean_dist = np.sqrt(np.dot(diff.T, diff))
                    min_dist = min(min_dist, euclidean_dist)

                wss += min_dist

        if self.__display:
            print(f"Within cluster sum of squares: {wss}")

        return wss

    def rand_index(self):
        if self.__labels is None:
            raise AttributeError("Actual labels are not available")

        lables_mapping = {}

        for cluster_label in self.__clusters_labels:
            actual_labels = []

            for i, predicted_label in enumerate(self.__predicted_labels):
                if predicted_label == cluster_label:
                    actual_labels.append(self.__labels[i])

            most_freq = max(set(actual_labels), key=actual_labels.count)
            lables_mapping[cluster_label] = [most_freq, actual_labels.count(most_freq)]

        lables_mapping = sorted(
            lables_mapping.items(), key=lambda x: x[1][1], reverse=True
        )

        agreeing_pairs_no = 0
        max_value = min(len(set(self.__labels)), len(self.__clusters_labels))

        for index in range(max_value):
            agreeing_pairs_no += lables_mapping[index][1][1]

        rand_index = agreeing_pairs_no / len(self.__labels)

        if self.__display:
            print(f"Rand index: {rand_index}")

        return rand_index

    def purity(self):
        if self.__labels is None:
            raise AttributeError("Actual labels are not available")

        purity = 0

        for cluster_label in self.__clusters_labels:
            actual_labels = []

            for i, predicted_label in enumerate(self.__predicted_labels):
                if predicted_label == cluster_label:
                    actual_labels.append(self.__labels[i])

            most_freq = max(set(actual_labels), key=actual_labels.count)
            purity += actual_labels.count(most_freq)

        purity /= len(self.__labels)

        if self.__display:
            print(f"Purity: {purity}")

        return purity

    def get_score(self, score_name):
        if score_name == "wss":
            return self.within_cluster_variation()
        elif score_name == "rand index":
            return self.rand_index()
        elif score_name == "purity":
            return self.purity()
