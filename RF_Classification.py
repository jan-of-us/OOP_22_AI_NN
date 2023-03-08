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
    data_obj = Classification_Data(data=data, trees=100, test_size=0.5, x_labels=["Atr1"])
    # Create classifier class
    filename = 'model.sav'
    #data_obj.model = pickle.load(open(filename, 'rb'))

    classifier = RF_Classification(data_obj)
    #pickle.dump(data_obj.model, open(filename, 'wb'))

    plt.show()
    print(data_obj.result_string)


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
        self.predictions = None

        self.run_classifier(data_obj)



    def run_classifier(self, data_obj):
        # train the model
        if data_obj.model is not None and isinstance(data_obj.model, RandomForestClassifier):
            self.model = data_obj.model
            print("Model loaded")
        else:
            self.model = self.train_model()
            data_obj.model = self.model
            print("Model created")

        try:
            # make predictions
            self.predictions = self.model.predict(self.x_test)
            # get evaluation
            data_obj.accuracy_score = accuracy_score(self.y_test, self.predictions)
            data_obj.result_string = f"The random forest classifier has a {accuracy_score(self.y_train, self.model.predict(self.x_train)):.2%} accuracy on the training data.\n"
            data_obj.result_string += f"The random forest classifier has a {data_obj.accuracy_score:.2%} accuracy on the testing data.\n"
            data_obj.result_string += super().evaluate(self.y_test, self.predictions)
            data_obj.feature_importance_dict = dict(zip(self.x_test.columns, self.model.feature_importances_))
            self.plot(data_obj)
        except ValueError:
            data_obj.result_string = "The loaded model does not match the set parameters, please try again!"

    def train_model(self):
        forest = RandomForestClassifier(n_estimators=self.n)
        forest.fit(self.evidence, self.labels)
        return forest


    def __str__(self):
        return "This method implements the random forest classification."


    def plot(self, data_obj):
        data_obj.confusion_matrix = super().plot_confusion_matrix(y_test=self.y_test, predictions=self.predictions)
        # feature importance pie chart
        fig, ax = plt.subplots()
        ax.pie(data_obj.feature_importance_dict.values(), labels=data_obj.feature_importance_dict.keys())
        ax.axis('equal')
        data_obj.feature_importance = fig

if __name__ == "__main__":
    main()
