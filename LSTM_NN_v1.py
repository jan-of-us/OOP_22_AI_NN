import numpy as np
import pandas as pd
#from Regression_Data import NN_LSTMModelGUI
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class NN_LSTMModel:
    def __init__(self, train_data, target_col, val_data=None, alpha=0.01, window_length=720, batch_size=32, hidden_layers=3):
        """
        Initializes an instance of NN_LSTMModel.
        Args:
            train_data (pandas.DataFrame): Training data with features and target variable.
            target_col (str): Name of the target variable column in train_data.
            val_data (pandas.DataFrame, optional): Validation data with the same columns as train_data.
            alpha (float, optional): Learning rate of the optimizer. Default is 0.01.
            window_length (int, optional): Number of previous time steps to use as input for predicting the next time step. Default is 720 - 5 days of data.
            batch_size (int, optional): Number of samples per gradient update. Default is 32.
            hidden_layers (array of int, optional): Number of hidden layers. Default is 3.
        """
        self.train_data = train_data
        self.target_col = target_col
        self.val_data = val_data
        self.alpha = alpha
        self.window_length = window_length
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers

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
        if self.val_data is not None:
            val_scaled = scaler.transform(self.val_data)
            return train_scaled, val_scaled, scaler
        else:
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
        train_data_scaled, val_data_scaled, scaler = self.preprocess_data()
        X_train, y_train = self.create_dataset(train_data_scaled, self.window_length)
        model = tf.keras.Sequential()
        for layer in self.hidden_layers:
            model.add(tf.keras.layers.LSTM(layer, return_sequences=True, input_shape=(self.window_length, X_train.shape[2])))
            model.add(tf.keras.layers.LeakyReLU(NN_LSTMModel.alpha))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.Dense(self.train_data.shape[1]))
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        history = model.fit(X_train, y_train, epochs=50, batch_size=self.batch_size, validation_data=(val_data_scaled, None), verbose=1)
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
        if self.val_data is not None:
            val_pred_scaled = self.model.predict(val_data_scaled[:-self.window_length])
            val_pred = scaler.inverse_transform(val_pred_scaled)
            val_target = self.val_data[self.window_length:]
            plt.plot(val_target, label='Validation Actual')
            plt.plot(val_pred, label='Validation Predicted')
        plt.xlabel('Time')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.show()