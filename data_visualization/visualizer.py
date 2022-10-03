from scipy.cluster.hierarchy import dendrogram
import plotly.express as px
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


class Visualizer:
    colors = ['#DFFF00', '#FFBF00', '#FF7F50', '#DE3163', '#9FE2BF',
              '#40E0D0', '#6495ED', '#CCCCFF', '#008000', '#000080']

    @staticmethod
    def plot_dataset_2d(dataset, dataset_name):
        title = f'2D representation of {dataset_name.capitalize()} dataset'

        Visualizer.__plot(dataset, title)

    @staticmethod
    def plot_predictions_dataset_2d(dataset, dataset_name, predicted_classes):
        title = f'Predictions for {dataset_name.capitalize()} dataset'
        dataset['class'] = predicted_classes

        Visualizer.__plot(dataset, title)

    @classmethod
    def __plot(cls, dataset, title):
        fig = px.scatter(dataset, x='first dimension', y='second dimension',
                         color='class', title=title, color_continuous_scale=cls.colors)
        fig.update_coloraxes(showscale=False)
        fig.show()

    @classmethod
    def plot_custering_evolution(cls, clusters_evolution, title, agglomerative=False):
        classes = set(clusters_evolution['class'].values)

        if agglomerative:
            fig = px.scatter(clusters_evolution, x="first dimension", y="second dimension", title=title.capitalize(),
                            animation_frame="iteration", color='class', color_continuous_scale=px.colors.cyclical.Twilight)
        else:
            fig = px.scatter(clusters_evolution, x="first dimension", y="second dimension", title=title,
                animation_frame="iteration", color='class', color_discrete_map=dict(zip(classes, cls.colors)))

        iter_no = clusters_evolution['iteration'].max()
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 10000 // (
            iter_no + 1)
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1000 // (
            iter_no + 1)

        fig.show()

    @classmethod
    def plot_dendrogram(cls, linkage_matrix, dataset_name):
        fig, ax = plt.subplots(1, 1)
        dendrogram(linkage_matrix, ax=ax)
        ax.set_xlabel('indices of points')
        ax.set_ylabel('distances between clusters')
        ax.set_title(f'Dendrogram of {dataset_name} data set')
        plt.show()
