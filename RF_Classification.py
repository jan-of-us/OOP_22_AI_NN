import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Classification import Classification
from Classification_Data import Classification_Data
from sklearn.metrics import accuracy_score

def main():
    # import test data
    data = pd.read_csv("Data/divorce.csv", delimiter=";")
    data_obj = Classification_Data(data=data, trees=10000)
    # Create classifier class
    filename = 'model.sav'
    #data_obj.model = pickle.load(open(filename, 'rb'))

    classifier = RF_Classification(data_obj)
    #pickle.dump(data_obj.model, open(filename, 'wb'))

    plt.show()
    print(data_obj.accuracy_score)


class RF_Classification(Classification):
    """
    RandomForest-classification.
    :param evidence: array of evidence, int or float,
    :param labels: list of labels int
    :param test_size: Size of testing data 0-1, default: 0.2 float,
    :param k: trees k=100
    :return: prints evaluation to terminal
    """
    def __init__(self, data_obj: Classification_Data):
        super().__init__(data_obj)

        self.n = data_obj.trees
        self.sensitivity, self.specificity, self.predictions = int(), int(), None

        self.run_classifier(data_obj)
        data_obj.feature_importance_dict = dict(zip(self.x_test.columns, self.model.feature_importances_))
        self.plot(data_obj)
        data_obj.accuracy_score = accuracy_score(self.y_test, self.predictions)

    def run_classifier(self, data_obj):
        # train the model
        if data_obj.model is not None and isinstance(data_obj.model, RandomForestClassifier):
            self.model = data_obj.model
            print("Model loaded")
        else:
            self.model = self.train_model()
            data_obj.model = self.model
            print("Model created")

        # make predictions
        self.predictions = self.model.predict(self.x_test)
        # get evaluation
        self.evaluate()

        # Print results
        print(f"Correct: {(self.y_test == self.predictions).sum()}")
        print(f"Incorrect: {(self.y_test != self.predictions).sum()}")
        print(f"True Positive Rate: {100 * self.sensitivity:.2f}%")
        print(f"True Negative Rate: {100 * self.specificity:.2f}%")
        print(dict(zip(self.x_test.columns, self.model.feature_importances_)))

    def train_model(self):
        forest = RandomForestClassifier(n_estimators=self.n)
        forest.fit(self.evidence, self.labels)
        return forest

    def evaluate(self):
        pos_label = 0
        neg_label = 0
        corr_pos_pred = 0
        corr_neg_pred = 0
        true_labels = self.y_test.to_numpy()
        for i in range(true_labels.shape[0]):
            # specificity
            if true_labels[i] == 0:
                neg_label += 1
                if self.predictions[i] == 0:
                    corr_neg_pred += 1
            # sensitivity
            if true_labels[i] == 1:
                pos_label += 1
                if self.predictions[i] == 1:
                    corr_pos_pred += 1
        self.sensitivity = corr_pos_pred / float(pos_label)
        self.specificity = corr_neg_pred / float(neg_label)

    def __str__(self):
        return "This method implements the random forest classification."

    def print_results(self):
        results = f"Correct: {(self.y_test == self.predictions).sum()}\n"\
                    f"Incorrect: {(self.y_test != self.predictions).sum()}\n"\
                    f"True Positive Rate: {100 * self.sensitivity:.2f}%\n"\
                    f"True Negative Rate: {100 * self.specificity:.2f}%"
        return results

    def plot(self, data_obj):
        data_obj.confusion_matrix = super().plot_confusion_matrix(y_test=self.y_test, predictions=self.predictions)
        # feature importance pie chart
        fig, ax = plt.subplots()
        ax.pie(data_obj.feature_importance_dict.values(), labels=data_obj.feature_importance_dict.keys())
        ax.axis('equal')
        data_obj.feature_importance = fig

if __name__ == "__main__":
    main()
