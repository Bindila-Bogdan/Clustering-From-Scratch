import sys

sys.path.append("../data_preprocessing")
sys.path.append("../clustering_algorithms")
from data_loader import DataLoader
from kmeans import KMeans
from scorer import Scorer
from sklearn.cluster import KMeans as KMeansScikit


def test_kmeans_wss():
    data_loader = DataLoader("iris")
    coordinates, labels = data_loader.data

    wss_manually_computed = sys.maxsize

    for _ in range(10):
        kmeans = KMeans(coordinates, labels, n_clusters=3, max_iter=100)
        kmeans.fit_transform()
        scorer = Scorer(coordinates, kmeans.predicted_labels, labels, False)
        wss_manually_computed = min(wss_manually_computed, scorer.get_score("wss"))

    kmeans_scikit = KMeansScikit(n_clusters=3, max_iter=100, n_init=10)
    kmeans_scikit.fit_transform(coordinates, labels)
    scorer = Scorer(coordinates, kmeans_scikit.labels_, labels, False)
    scikit_wss_manually_computed = scorer.get_score("wss")

    metrics_ratio = wss_manually_computed / scikit_wss_manually_computed
    print(f"WSS KMeans from scratch: {wss_manually_computed}")
    print(f"WSS Scikit-Learn KMeans: {scikit_wss_manually_computed}")

    assert 0.99 < metrics_ratio < 1.01


if __name__ == "__main__":
    test_kmeans_wss()