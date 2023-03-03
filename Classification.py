from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from Classification_Data import Classification_Data


class Classification:
    """
    Class for all classification methods
    """
    def __init__(self, data_obj: Classification_Data):
        """
        :param data: dataframe containing the dataset
        :param test_size: share of data that is used for testing. Default: 0.2
        """
        # initialize necessary variables
        self.evidence, self.labels, self.model = pd.DataFrame, pd.DataFrame, None
        self.data = data_obj.data
        self.test_size = data_obj.test_size

        # split the dataset into evidence and labels
        self.split_evidence_labels(data_obj)

        # split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.evidence, self.labels, test_size=self.test_size
        )

    def split_evidence_labels(self, data_obj):
        """
        Splits given dataset into evidence and labels, requires labels to be last column of dataframe
        """
        if data_obj.x_labels is None:
            if data_obj.y_label is None:
                self.evidence = self.data.iloc[:, :-1]
            else:
                self.evidence = self.data
                self.evidence.drop(labels=data_obj.y_label, axis='columns')
        else:
            self.evidence = self.data[data_obj.x_labels]
        if data_obj.y_label is None:
            self.labels = self.data[self.data.columns[-1]]
        else:
            self.labels = self.data[data_obj.y_label]
        print(self.evidence)
        print(self.labels)

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
        Generates a confusion matrix with given labels and predictions
        :param y_test: real labels
        :param predictions: predicted labels
        :return: matplotlib subplot
        """
        conf_matrix = confusion_matrix(y_test, predictions)
        conf_matrix = pd.DataFrame(conf_matrix)
        fig, ax = plt.subplots()
        ax = sn.heatmap(conf_matrix, annot=True)
        return fig
