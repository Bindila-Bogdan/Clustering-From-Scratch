import sys

sys.path.append("../data_preprocessing")
from data_loader import DataLoader
from clustering import Clustering
from optimizer import Optimizer


class ClusteringRunner:
    @staticmethod
    def get_best_optimized_algorithm(dataset_name, metric="wss"):
        data_loader = DataLoader(dataset_name)
        algorithms = ["kmeans", "dbscan", "gaussian mixture"]

        if metric == "wss":
            best_score = sys.maxsize
        else:
            best_score = -sys.maxsize

        best_algorithm = None
        best_algorithm_name = None

        for algorithm in algorithms:
            optimizer = Optimizer(
                algorithm, data_loader.data[0], data_loader.data[1], metric
            )
            algorithm, score = optimizer.optimize()

            if (score < best_score and metric == "wss") or (
                score > best_score and metric != "wss"
            ):
                best_score = score
                best_algorithm = algorithm
                best_algorithm_name = algorithm

        print(f"Best algorithm: {best_algorithm_name}\n{metric}: {best_score}")
        clustering = Clustering(dataset_name, best_algorithm_name, best_algorithm)
        clustering.visualize_result()

    @staticmethod
    def test_algorithms():
        algorithm_dataset = [
            ("kmeans", "diamonds"),
            ("dbscan", "smile"),
            ("agglomerative", "iris"),
            ("gaussian mixture", "cure"),
        ]

        for algorithm_name, dataset_name in algorithm_dataset:
            clustering = Clustering(dataset_name, algorithm_name)
            clustering.train()
            clustering.visualize_result()


if __name__ == "__main__":
    ClusteringRunner.get_best_optimized_algorithm("diamonds")
    ClusteringRunner.test_algorithms()
