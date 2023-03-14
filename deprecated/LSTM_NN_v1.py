import numpy as np
from AI_Regression.Regression import Regression
import pandas as pd
from AI_Regression.Regression_Data import Regression_Data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class NN_LSTMModel(Regression):
    def __init__(self, data_obj: Regression_Data):
        """
        Initializes an instance of NN_LSTMModel.
        :param data_obj: Regression Data object
        """
        super().__init__(data_obj)
        self.train_data = data_obj.data
        self.target_col = data_obj.y_label
        self.alpha = data_obj.alpha
        self.window_length = data_obj.window_length
        self.batch_size = data_obj.batch_size
        self.hidden_layers = data_obj.hidden_layers

        self.preprocess_data()
        self.create_dataset(self.data.to_numpy(), self.window_length)
        self.fit()
        self.predict(data=self.data)
        self.plot_results()

    def preprocess_data(self):
        """
        Scales the train and validation data to the range of 0 to 1.
        Returns:
            train_scaled (numpy.ndarray): Scaled training data.
            val_scaled (numpy.ndarray): Scaled validation data.
            scaler (sklearn.preprocessing.MinMaxScaler): Scaler used to scale the data.
        """
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(self.train_data)

        return train_scaled, scaler

    def create_dataset(self, data, window_length):
        """
        Creates a dataset for training or prediction by sliding a window of size window_length over the input data.
        Args:
            data (numpy.ndarray): Scaled input data.
            window_length (int): Number of previous time steps to use as input for predicting the next time step.
        Returns:
            X (numpy.ndarray): Input data with shape (n_samples, window_length, n_features).
            y (numpy.ndarray): Target data with shape (n_samples, n_features).
        """
        X, y = [], []
        for i in range(window_length, len(data)):
            X.append(data[i-window_length:i, :])
            y.append(data[i, :])
        X, y = np.array(X), np.array(y)
        return X, y

    def fit(self):
        """
        Trains the LSTM model using the training data and validation data (if available).
        """
        train_data_scaled, scaler = self.preprocess_data()
        X_train, y_train = self.create_dataset(train_data_scaled, self.window_length)
        model = tf.keras.Sequential()
        for layer in self.hidden_layers:
            model.add(tf.keras.layers.LSTM(layer, return_sequences=True, input_shape=(self.window_length, X_train.shape[2])))
            model.add(tf.keras.layers.LeakyReLU(self.alpha))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(self.train_data.shape[1]))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        history = model.fit(X_train, y_train, epochs=50, batch_size=self.batch_size, verbose=1)
        self.model = model
        self.scaler = scaler
        self.history = history

    def predict(self, data):
        """
        Predicts the target variable for the given input data.
        """
        data_scaled = self.scaler.transform(data)
        X, _ = self.create_dataset(data_scaled, self.window_length)
        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        return y_pred

    def plot_results(self):
        """
        Plots the actual vs. predicted values for the training and validation data.
        """
        train_data_scaled, val_data_scaled, scaler = self.preprocess_data()
        train_pred_scaled = self.model.predict(train_data_scaled[:-self.window_length])
        train_pred = scaler.inverse_transform(train_pred_scaled)
        train_target = self.train_data[self.window_length:]
        plt.plot(train_target, label='Actual')
        plt.plot(train_pred, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.show()

def main():
    # import test data
    data = pd.read_csv("Data/energydata_complete.csv", sep=",", parse_dates=["date"])

    # data preprocessing
    #data.dropna()
    #data['year'] = data['date'].dt.year
    #data['month'] = data['date'].dt.month
    #data['day'] = data['date'].dt.day
    #data['hour'] = data['date'].dt.hour
    #data['minute'] = data['date'].dt.minute
    data = data.drop(columns=["date"])


    # Create Data Class, start index and n_values atm only used for plotting, training and prediction done on all data
    data_obj = Regression_Data(data=data, y_label="lights", n_values=50)

    # Create classifier class
    regressor = NN_LSTMModel(data_obj)
    plt.show()
    print(data_obj.result_string)


if __name__ == "__main__":
    main()