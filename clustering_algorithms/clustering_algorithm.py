from abc import ABC, abstractmethod

class ClusteringAlgorithm(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass

    # TODO: impute the scoring method
    @abstractmethod
    def score(self, scorer):
        pass
