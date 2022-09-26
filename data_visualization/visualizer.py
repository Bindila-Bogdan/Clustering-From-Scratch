import plotly.express as px


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
    def plot_custering_evolution(cls, clusters_evolution, title):
        classes = set(clusters_evolution['class'].values)

        fig = px.scatter(clusters_evolution, x="first dimension", y="second dimension", title=title,
                         animation_frame="iteration", color='class', color_discrete_map=dict(zip(classes, cls.colors)))
        #fig.update_coloraxes(showscale=False)

        iter_no = clusters_evolution['iteration'].max()
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 10000 // (iter_no + 1)
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1000 // (iter_no + 1)

        fig.show()
