import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DataLoader:
    def __init__(self, dataset_name) -> None:
        self.__dataset_name = dataset_name
        self.__data = self.__standardize_data(self.__load_data())
        self.__data_2d = self.__prepare_data()

    def __standardize_data(self, data):
        label = data[data.columns[-1]]

        std_scaler = StandardScaler()
        std_data_ = std_scaler.fit_transform(data.drop(data.columns[-1], axis=1))

        std_data = pd.DataFrame(std_data_)
        std_data["class"] = label
        del std_data_

        return std_data

    def __reduce_dimensionality(self, data):
        label = data[data.columns[-1]]

        pca = PCA(n_components=2)
        reduced_data_ = pca.fit_transform(data.drop(data.columns[-1], axis=1))

        reduced_data = pd.DataFrame(reduced_data_)
        reduced_data["class"] = label
        del reduced_data_

        return reduced_data

    def __load_data(self):
        data_path = f"../datasets/{self.__dataset_name}.csv"
        data = pd.read_csv(data_path, low_memory=False, index_col=0)
        data = data.reset_index()

        return data

    def __prepare_data(self):
        if self.__data.shape[1] > 3:
            data_2d = self.__reduce_dimensionality(self.__data)
        else:
            data_2d = self.__standardize_data(self.__data)

        data_2d.columns = ["first dimension", "second dimension", "class"]

        return data_2d

    def display_dataset(self):
        print(f"{self.__dataset_name} dataset")

        if self.__data.shape[1] > 3:
            print(f"Original version: {self.__data}\n2D version: {self.__data_2d}")
        else:
            print(self.__data)

    @property
    def data(self):
        labels = np.array(self.__data[self.__data.columns[-1]].values)
        coordinates = self.__data[self.__data.columns[:-1]].values

        return coordinates, labels

    @property
    def data_2d(self):
        return self.__data_2d

    @property
    def dataset_name(self):
        return self.__dataset_name
