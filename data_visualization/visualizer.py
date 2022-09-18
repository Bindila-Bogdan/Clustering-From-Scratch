import plotly.express as px


class Visualizer:
    @staticmethod
    def plot_dataset_2d(dataset, dataset_name, predicted_classes=None):
        title = f'2D representation of {dataset_name.capitalize()} dataset'

        fig = px.scatter(dataset, x='first dimension', y='second dimension',
                         color='class', symbol='class', title=title, color_discrete_sequence=px.colors.sequential.Purples_r)
        fig.update_coloraxes(showscale=False)
        fig.show()

    @staticmethod    
    def plot_prediction_dataset_2d(dataset, dataset_name, predicted_classes):
        title = f'Predictions for {dataset_name.capitalize()} dataset'
        print(type(predicted_classes))
        dataset['class'] = predicted_classes

        fig = px.scatter(dataset, x='first dimension', y='second dimension',
                         color='class', symbol='class', title=title, color_discrete_sequence=px.colors.sequential.Purples_r)
        fig.update_coloraxes(showscale=False)
        fig.show()