import numpy as np
import pandas as pd
import plotly.express as px
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
        std_data['class'] = label
        del std_data_

        return std_data

    def __reduce_dimensionality(self, data):
        label = data[data.columns[-1]]

        pca = PCA(n_components=2)
        reduced_data_ = pca.fit_transform(data.drop(data.columns[-1], axis=1))

        reduced_data = pd.DataFrame(reduced_data_)
        reduced_data['class'] = label
        del reduced_data_

        return reduced_data

    def __load_data(self):
        data_path = f'../datasets/{self.__dataset_name}.csv'
        data = pd.read_csv(data_path, low_memory=False, index_col=0)
        data = data.reset_index()

        return data

    def __prepare_data(self):
        if self.__data.shape[1] > 3:
            data_2d = self.__reduce_dimensionality(self.__data)
        else:
            data_2d = self.__data

        data_2d_std = self.__standardize_data(data_2d)
        data_2d_std.columns = ['first dimension', 'second dimension', 'class']
        del data_2d

        return data_2d_std

    def display_dataset(self):
        print(f'{self.__dataset_name} dataset')

        if self.__data.shape[1] > 3:
            print(
                f'Original version: {self.__data}\n2D version: {self.__data_2d}')
        else:
            print(self.__data)

    def plot_dataset_2d(self):
        if self.__data.shape[1] > 3:
            title = f'2D representation of {self.__dataset_name.capitalize()} dataset'
        else:
            title = f'{self.__dataset_name.capitalize()} dataset'

        print(self.__data_2d)
        fig = px.scatter(self.__data_2d, x='first dimension', y='second dimension',
                         color='class', symbol='class', title=title, color_discrete_sequence= px.colors.sequential.Purples_r)
        fig.update_coloraxes(showscale=False)
        fig.show()

    def get_data(self):
        labels = np.array(self.__data[self.__data_2d.columns[-1]].values)
        coordinates = self.__data[self.__data.columns[:-1]].values

        return coordinates, labels
