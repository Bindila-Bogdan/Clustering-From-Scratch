import plotly.express as px


class Visualizer:
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
                         color='class', symbol='class', title=title,
                         color_discrete_sequence=px.colors.sequential.Rainbow)
        fig.update_coloraxes(showscale=False)
        fig.show()
