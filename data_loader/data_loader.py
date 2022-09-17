import pandas as pd


class DataLoader:
    def __init__(self, dataset_name) -> None:
        self.__dataset_name = dataset_name
        self.__data = self.__load_data()

    def __load_data(self):
        data_path = f'../dataset/{self.__dataset_name}.csv'
        data = pd.read_csv(data_path, low_memory=False, index_col=0)

        return data

    def display_dataset(self):
        print(f'{self.__dataset_name} dataset')
        print(self.__data)

    def plot_dataset(self):
        pass
