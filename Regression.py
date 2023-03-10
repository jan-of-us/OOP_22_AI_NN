from sklearn.model_selection import train_test_split
import pandas as pd
from Regression_Data import Regression_Data


class Regression:
    """
    Class for all regression methods
    """
    def __init__(self, data_obj: Regression_Data):
        """
        :param data: dataframe containing the dataset
        :param test_size: share of data that is used for testing. Default: 0.2
        """
        # initialize necessary variables
        self.features, self.target, self.model = pd.DataFrame, pd.DataFrame, None
        self.data = data_obj.data
        self.test_size = data_obj.test_size

        # split the dataset into evidence and labels
        self.split_data(data_obj)

        # split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=self.test_size
        )

    def split_data(self, data_obj):
        """
        Splits given dataset into evidence and labels, requires labels to be last column of dataframe
        """
        if data_obj.x_labels is None:
            if data_obj.y_label is None:
                self.features = self.data.iloc[:, :-1]
            else:
                self.features = self.data.drop(columns=[data_obj.y_label])
        else:
            self.features = self.data[data_obj.x_labels]
        if data_obj.y_label is None:
            self.target = self.data[self.data.columns[-1]]
            data_obj.y_label = self.data.columns[-1]
        else:
            self.target = self.data[data_obj.y_label]
        print(self.features)
        print(self.target)

    def __str__(self):
        """
        Returns a string with infos about the used methods and the achieved results
        :return:
        """
        return f"This is a Classification Superclass"

    def plot(self, data_obj):
        """

        :return:
        """
        raise NotImplementedError
