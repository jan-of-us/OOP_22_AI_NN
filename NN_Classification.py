import tensorflow as tf
from data_import import import_csv
from Classification import Classification
import matplotlib.pyplot as plt


class NN_Classification(Classification):
    def __init__(self, data, test_size=0.2, epochs=10):
        super().__init__(data, test_size=0.2)
        self.epochs = epochs

        # get the number of output categories from the dataset
        self.output_categories = self.y_test.nunique()

        # turn labels into matrix format, needed for categorical_crossentropy
        self.y_train = tf.keras.utils.to_categorical(self.y_train)
        self.y_test = tf.keras.utils.to_categorical(self.y_test)

        # create the neural network
        self.get_model()

        # train the neural network
        self.history = self.model.fit(self.x_train, self.y_train, validation_split=0.2, epochs=self.epochs)

        # testing the network on the testing data
        self.model.evaluate(self.x_test, self.y_test, verbose=2)

    def get_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.evidence.shape[1])),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    def plot(self, param=0):
        """
        Plot the results/history from the classification
        :param param: Which plots to display: 0=all, 1=accuracy, 2=loss
        :return:
        """
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
        plt.show()


def main(file):
    data = import_csv(file)
    classifier = NN_Classification(data)
    classifier.plot(1)


if __name__ == "__main__":
    main("Data/divorce.csv")
