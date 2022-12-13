import tensorflow as tf
from data_import import import_csv
from Classification import Classification
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import pandas as pd
import seaborn as sn


class NN_Classification(Classification):
    def __init__(self, data, test_size=0.2, epochs=10):
        super().__init__(data, test_size=0.2)
        self.epochs = epochs

        # get the number of output categories from the dataset
        self.output_categories = self.y_test.nunique()

        # create the neural network
        self.get_model()

        # train the neural network
        self.history = self.model.fit(self.x_train, self.y_train, validation_split=0.2, epochs=self.epochs)

        # predictions for confusion matrix
        self.predictions = self.model.predict(self.x_test)

        # testing the network on the testing data
        self.model.evaluate(self.x_test, self.y_test, verbose=2)

    def get_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.evidence.shape[1])),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def plot(self, param=0):
        """
        Plot the results/history from the classification
        :param param: Which plots to display: 0=all, 1=accuracy, 2=loss, 3=confusion-matrix
        :return:
        """
        if param != 3:
            fig = plt.figure
            if param in [0, 1]:
                plt.plot(self.history.history["accuracy"])
                plt.plot(self.history.history["val_accuracy"])
            if param in [0, 2]:
                plt.plot(self.history.history["loss"])
                plt.plot(self.history.history["val_loss"])
            plt.legend(["training accuracy", "testing accuracy", "train-loss", "test-loss"], loc="best")
            plt.xlabel("epoch")
            plt.ylabel("accuracy/loss")
            plt.title("model accuracy & loss")
            plt.grid()

            return fig

        # convert predictions from percentages to labels
        self.predictions = tf.argmax(self.predictions, 1)
        self.conf_matrix = tf.math.confusion_matrix(self.y_test, self.predictions, num_classes=2)
        self.conf_matrix = pd.DataFrame(self.conf_matrix)
        fig, ax = plt.subplots()
        plt.figure(figsize=(4, 4))
        ax = sn.heatmap(self.conf_matrix, annot=True)
        return fig


def main(file):
    data = import_csv(file)
    classifier = NN_Classification(data)
    fig = classifier.plot(3)
    plt.show()


if __name__ == "__main__":
    main("Data/divorce.csv")
