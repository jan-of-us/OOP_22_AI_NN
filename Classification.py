from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from Classification_Data import Classification_Data
from sklearn.preprocessing import LabelEncoder


class Classification:
    """
    Class for all classification methods
    """
    def __init__(self, data_obj: Classification_Data):
        """
        :param data_obj: Classification_Data object
        """
        # initialize necessary variables
        self.evidence, self.labels, self.model = pd.DataFrame, pd.DataFrame, None
        self.data = data_obj.data
        data_obj.data = self.encode()
        self.test_size = data_obj.test_size

        # split the dataset into evidence and labels
        self.split_evidence_labels(data_obj)

        # split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.evidence, self.labels, test_size=self.test_size
        )

    def encode(self):
        """
        Encodes variables that are not integer or float format
        :return: converted dataframe
        """
        label_encoder = LabelEncoder()
        for value in self.data.select_dtypes(include=["object"]).columns.values:
            self.data[value] = label_encoder.fit_transform(self.data[value])
        return self.data


    def split_evidence_labels(self, data_obj):
        """
        Splits given dataset into evidence and labels
        """
        if data_obj.x_labels is None:
            if data_obj.y_label is None:
                self.evidence = self.data.iloc[:, :-1]
            else:
                self.evidence = self.data.drop(columns=[data_obj.y_label])
        else:
            self.evidence = self.data[data_obj.x_labels]
        if data_obj.y_label is None:
            self.labels = self.data[self.data.columns[-1]]
            data_obj.y_label = self.data.columns[-1]
        else:
            self.labels = self.data[data_obj.y_label].subtract(self.data[data_obj.y_label].min())
        print(self.evidence)
        print(self.labels)


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

    @staticmethod
    def evaluate(y_test, predictions):
        """
        Evaluate Predictions
        :param y_test: real labels
        :param predictions: predicted labels
        :return: String with Results
        """
        pos_label = 0
        neg_label = 0
        corr_pos_pred = 0
        corr_neg_pred = 0
        true_labels = y_test.to_numpy()
        for i in range(true_labels.shape[0]):
            # specificity
            if true_labels[i] == 0:
                neg_label += 1
                if predictions[i] == 0:
                    corr_neg_pred += 1
            # sensitivity
            if true_labels[i] == 1:
                pos_label += 1
                if predictions[i] == 1:
                    corr_pos_pred += 1
        # sensitivity = corr_pos_pred / float(pos_label)
        # specificity = corr_neg_pred / float(neg_label)
        results = f"Detailed results on testing set:\n"\
                  f"Correct: {(y_test == predictions).sum()}\n" \
                  f"Incorrect: {(y_test != predictions).sum()}\n" \
                #   f"True Positive Rate: {100 * sensitivity:.2f}%\n" \
                #   f"True Negative Rate: {100 * specificity:.2f}%"
        return results
