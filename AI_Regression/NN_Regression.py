import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from AI_Regression.Regression import Regression
from AI_Regression.Regression_Data import Regression_Data


class NN_Regression(Regression):
    """
    RandomForest-classification.
    :param data_obj: Regression_Data object
    :return: data_obj with filled result variables
    """
    def __init__(self, data_obj: Regression_Data):
        super().__init__(data_obj)
        self.hidden_layers = data_obj.hidden_layers
        self.window_length = data_obj.window_length
        self.alpha = data_obj.alpha
        self.batch_size = data_obj.batch_size
        self.sensitivity, self.specificity, self.predictions = int(), int(), None
        self.run_classifier(data_obj)

    def run_classifier(self, data_obj):
        # train the model
        if data_obj.model is not None and isinstance(data_obj.model, tf.keras.models.Sequential):
            self.model = data_obj.model
            print("Model loaded")
        else:
            self.model = self.train_model(data_obj)
            data_obj.model = self.model
            print("Model created")

        try:
            # make predictions
            self.predictions = pd.DataFrame(self.model.predict(self.x_test))

            # get evaluation
            self.evaluate(data_obj)

            # Print results
            self.print_results(data_obj)
            # Create plots
            self.plot(data_obj)
        except ValueError:
            data_obj.result_string = "The loaded model does not match the set parameters, please try again!"
    def train_model(self, data_obj):
        model = tf.keras.Sequential()
        for layer in data_obj.hidden_layers:
            model.add(tf.keras.layers.Dense(layer, activation=data_obj.activation_func))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer="adam", metrics=['mse'])
        self.history = model.fit(x=self.x_train, y=self.y_train, epochs=data_obj.training_epochs, validation_split=0.2)
        model.evaluate(x=self.x_test, y=self.y_test)
        #model.save("model.h5")
        return model


    def evaluate(self, data_obj):
        data_obj.r2_score = r2_score(self.y_test, self.predictions)
        data_obj.mean_abs_error = mean_absolute_error(self.y_test, self.predictions)
        data_obj.mean_sqr_error = mean_squared_error(self.y_test, self.predictions)


    def __str__(self):
        return "This method implements the neural network regression."

    def print_results(self, data_obj):
        data_obj.result_string += f"The classifiers R2_Score is {data_obj.r2_score}.\n\n" \
                                 f"The mean abs. error is {data_obj.mean_abs_error}.\n\n" \
                                 f"The mean squared error is {data_obj.mean_sqr_error}."

    def plot(self, data_obj):
        # plot predictions
        super().plot_predictions(y_scaler=self.y_scaler, y_test=self.y_test, predictions=self.predictions,
                                 data_obj=data_obj)

        try:
            fig = plt.figure()
            plt.plot(self.history.history["mse"])
            plt.plot(self.history.history["val_mse"])
            plt.plot(self.history.history["loss"])
            plt.plot(self.history.history["val_loss"])
            plt.legend(["training accuracy", "testing accuracy", "train-loss", "test-loss"], loc="best")
            plt.xlabel("epoch")
            plt.ylabel("accuracy/loss")
            plt.title("model accuracy & loss")
            plt.grid()
            data_obj.loss_per_epoch = fig
        except AttributeError:
            # Loaded model has no history
            data_obj.loss_per_epoch = None
        except KeyError:
            # Loaded model has no history
            data_obj.loss_per_epoch = None


def main():
    # import test data
    data = pd.read_csv("./Data/energydata_complete.csv")

    # Create Data Class, start index and n_values atm only used for plotting, training and prediction done on all data
    data_obj = Regression_Data(data=data, x_labels=["T1"], y_label="lights", scale=True, hidden_layers=[32, 64, 32], training_epochs=50)

    filename = 'model.h5'
    data_obj.model = tf.keras.models.load_model(filename)

    # Create classifier class
    regressor = NN_Regression(data_obj)
    #data_obj.model.save(filename)
    plt.show()
    print(data_obj.result_string)



if __name__ == "__main__":
    main()
