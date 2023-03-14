import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from AI_Regression.Regression import Regression
from AI_Regression.Regression_Data import Regression_Data
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator


def main():
    # import test data
    data = pd.read_csv("Data/energydata_complete.csv", sep=",", parse_dates=["date"])

    # data preprocessing
    data.dropna()
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['hour'] = data['date'].dt.hour
    data['minute'] = data['date'].dt.minute
    data = data.drop(columns=["date"])


    # Create Data Class, start index and n_values atm only used for plotting, training and prediction done on all data
    data_obj = Regression_Data(data=data, y_label="lights",)
    print(data_obj)

    filename = 'model.h5'
    data_obj.model = tf.keras.models.load_model(filename)

    # Create classifier class
    regressor = NN_Regression_LSTM(data_obj)
    #data_obj.model.save(filename)
    plt.show()
    print(data_obj.result_string)


class NN_Regression_LSTM(Regression):
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
        self.train_generator = TimeseriesGenerator(self.x_train.to_numpy(), self.y_train.to_numpy(),
                                                   length=self.window_length, sampling_rate=1,
                                                   batch_size=self.batch_size)
        self.test_generator = TimeseriesGenerator(self.x_test.to_numpy(), self.y_test.to_numpy(),
                                                  length=self.window_length, sampling_rate=1,
                                                  batch_size=self.batch_size)
        self.run_classifier(data_obj)
        print(self.predictions)

        self.plot(data_obj)

    def run_classifier(self, data_obj):
        # train the model
        if data_obj.model is not None and isinstance(data_obj.model, tf.keras.models.Sequential):
            self.model = data_obj.model
            print("Model loaded")
        else:
            self.model = self.train_model()
            data_obj.model = self.model
            print("Model created")

        # make predictions
        self.predictions = self.model.predict(self.test_generator)


        # get evaluation
        self.evaluate(data_obj)

        # Print results
        self.print_results(data_obj)


    def train_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(self.window_length, self.x_train.shape[1])))
        model.add(tf.keras.layers.LeakyReLU(self.alpha))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode=min)
        self.history = model.fit(self.train_generator, epochs=5, validation_split=0.2)
        model.evaluate(self.test_generator)
        #model.save("model.h5")
        return model


    def evaluate(self, data_obj):
        data_obj.r2_score = r2_score(self.y_test, self.predictions)
        #data_obj.mean_abs_error = mean_absolute_error(self.y_test, self.predictions)
        #data_obj.mean_sqr_error = mean_squared_error(self.y_test, self.predictions)
        #data_obj.feature_importance_dict = dict(zip(self.x_test.columns, self.model.feature_importances_))

    def __str__(self):
        return "This method implements the random forest classification."

    def print_results(self, data_obj):
        data_obj.result_string = f"The classifiers R2_Score is {data_obj.r2_score}.\n\n" \
                                 f"The mean abs. error is {data_obj.mean_abs_error}.\n\n" \
                                 f"The mean squared error is {data_obj.mean_sqr_error}."

    def plot(self, data_obj):
        fig, ax = plt.subplots()
        plt.plot(self.y_test.to_numpy()[0:data_obj.n_values], color='red', label='Real data')
        plt.plot(self.predictions[0][0:data_obj.n_values], color='blue', label='Predicted data')
        plt.title('Prediction')
        plt.legend()
        data_obj.prediction = fig





if __name__ == "__main__":
    main()
