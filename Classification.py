from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf


class Classification:
    """
    Class for all classification methods
    """
    def __init__(self, data, test_size=0.2):
        """
        :param data: dataframe containing the dataset
        :param test_size: share of data that is used for testing. Default: 0.2
        """
        self.evidence, self.labels, self.model = pd.DataFrame, pd.DataFrame, None
        self.data = data
        self.test_size = test_size

        self.split_evidence_labels()

        # split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.evidence, self.labels, test_size=self.test_size
        )

    def split_evidence_labels(self):
        """
        Splits given dataset into evidence and labels
        """
        self.labels = self.data[self.data.columns[-1]]
        self.evidence = self.data.iloc[:, :-1]

    def return_data(self):
        """
        Returns the data in it's original format but manipulated state
        :return: dataframe
        """
        raise NotImplementedError

    def __str__(self):
        """
        Returns a string with infos about the used methods and the achieved results
        :return:
        """
        raise NotImplementedError

    def plot(self):
        """

        :return:
        """
        raise NotImplementedError
