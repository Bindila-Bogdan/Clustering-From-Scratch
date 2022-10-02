import sys
from kmeans import KMeans
from scorer import Scorer


class Optimizer:
    def __init__(self, algorithm_name, coordinates, labels, optimization_metric) -> None:
        self.__algorithm_name = algorithm_name
        self.__coordinates = coordinates
        self.__labels = labels
        self.__optimization_metric = optimization_metric

    def __optimize_kmeans(self):
        max_score = -sys.maxsize
        best_kmeans = None
        clusters_no = len(set(self.__labels))
        distance_metrics = ['euclidean', 'manhattan',
                            'cosine_similarity', 'pearson_correlation']

        for distance_metric in distance_metrics:
            print(f'distance metric = {distance_metric}')
            best_kmeans_instance = None
            max_score_instance = -sys.maxsize
            best_predicted_labels = None

            for _ in range(10):
                kmeans = KMeans(self.__coordinates, self.__labels,
                                n_clusters=clusters_no, distance_type=distance_metric)
                kmeans.fit_transform()
                scorer = Scorer(self.__coordinates, kmeans.predicted_labels)
                wss = scorer.get_score('wss')

                if wss > max_score_instance:
                    max_score_instance = wss
                    best_kmeans_instance = wss
                    best_predicted_labels = kmeans.predicted_labels
                else:
                    del kmeans

                del scorer

            scorer = Scorer(self.__coordinates,
                            best_predicted_labels, self.__labels)
            score = scorer.get_score(self.__optimization_metric)

            if score > max_score:
                max_score = score
                best_kmeans = best_kmeans_instance
            else:
                del best_kmeans_instance

            del scorer

        print(kmeans)

        return best_kmeans

    def optimize(self):
        print(f'Optimizing {self.__algorithm_name}...')

        if self.__algorithm_name == 'kmeans':
            self.__optimize_kmeans()
        elif self.__algorithm_name == 'dbscan':
            self.__optimize_dbscan()
        elif self.__algorithm_name == 'agglomerative':
            self.__optimize_agglomerative()
        elif self.__algorithm_name == 'gaussian mixture':
            self.__optimize_gaussian_mixture()
