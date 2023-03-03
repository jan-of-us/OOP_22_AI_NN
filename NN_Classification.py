import keras.models
import tensorflow as tf
from data_import import import_csv
from Classification import Classification
import matplotlib.pyplot as plt
from Classification_Data import Classification_Data
import plotly.figure_factory as ff
import pandas as pd
import seaborn as sn



class NN_Classification(Classification):
    def __init__(self, data_obj: Classification_Data):
        super().__init__(data_obj)

        # get the number of output categories from the dataset
        self.output_categories = self.y_test.nunique()

        if data_obj.model is not None and isinstance(data_obj.model, tf.keras.models.Sequential):
            self.model = data_obj.model
            print("Model loaded")
        else:
            # create the neural network
            self.get_model(data_obj)
            # train the neural network
            self.history = self.model.fit(self.x_train, self.y_train, validation_split=0.2, epochs=data_obj.training_epochs)
            data_obj.model = self.model
            print("Model created")



        # predictions for confusion matrix
        self.predictions = self.model.predict(self.x_test)

        # testing the network on the testing data
        self.model.evaluate(self.x_test, self.y_test, verbose=2)

        self.plot(data_obj)

    def get_model(self, data_obj):
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.evidence.shape[1])),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.output_categories, activation='softmax')
        ])
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    def plot(self, data_obj):
        """
        Plot the results/history from the classification
        :param param: Which plots to display: 0=all, 1=accuracy, 2=loss, 3=confusion-matrix
        :return:
        """
        try:
            fig = plt.figure()
            plt.plot(self.history.history["accuracy"])
            plt.plot(self.history.history["val_accuracy"])
            plt.plot(self.history.history["loss"])
            plt.plot(self.history.history["val_loss"])
            plt.legend(["training accuracy", "testing accuracy", "train-loss", "test-loss"], loc="best")
            plt.xlabel("epoch")
            plt.ylabel("accuracy/loss")
            plt.title("model accuracy & loss")
            plt.grid()
            data_obj.accuracy_per_epoch = fig
        except AttributeError:
            data_obj.accuracy_per_epoch = None

        # convert predictions from percentages to labels
        conf_predictions = tf.argmax(self.predictions, 1)
        data_obj.confusion_matrix = Classification.plot_confusion_matrix(self.y_test, conf_predictions)



def main(file):
    data = pd.read_csv(file, delimiter=";")
    data_obj = Classification_Data(data=data)
    print(type(data_obj.model))
    filename = 'model.sav'
    data_obj.model = keras.models.load_model('keras_model')
    classifier = NN_Classification(data_obj)
    #data_obj.model.save('keras_model')
    plt.show()


if __name__ == "__main__":
    main("Data/divorce.csv")
