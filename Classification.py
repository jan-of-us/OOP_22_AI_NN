from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


class Classification:
    """
    Class for all classification methods
    """
    def __init__(self, data, test_size=0.2):
        """
        :param data: dataframe containing the dataset
        :param test_size: share of data that is used for testing. Default: 0.2
        """
        # initialize necessary variables
        self.evidence, self.labels, self.model = pd.DataFrame, pd.DataFrame, None
        self.data = data
        self.test_size = test_size

        # split the dataset into evidence and labels
        self.split_evidence_labels()

        # split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.evidence, self.labels, test_size=self.test_size
        )

    def split_evidence_labels(self):
        """
        Splits given dataset into evidence and labels, requires labels to be last column of dataframe
        """
        self.labels = self.data[self.data.columns[-1]]
        self.evidence = self.data.iloc[:, :-1]

    def return_data(self):
        """
        Returns the data in it's original format but manipulated state
        :return: dataframe
        """
        return self.data

    def __str__(self):
        """
        Returns a string with infos about the used methods and the achieved results
        :return:
        """
        return f"This is a Classification Superclass"

    @staticmethod
    def plot_confusion_matrix(y_test, predictions):
        """

        :return:
        """
        conf_matrix = confusion_matrix(y_test, predictions)
        conf_matrix = pd.DataFrame(conf_matrix)
        fig, ax = plt.subplots()
        ax = sn.heatmap(conf_matrix, annot=True)
        return fig
