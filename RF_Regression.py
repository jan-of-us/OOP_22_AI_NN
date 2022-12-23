from data_import import import_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from Regression import Regression
import pandas as pd


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



    # Create classifier class
    regressor = RF_Regression(data)


class RF_Regression(Regression):
    """
    RandomForest-classification.
    :param evidence: array of evidence, int or float,
    :param labels: list of labels int
    :param test_size: Size of testing data 0-1, default: 0.2 float,
    :param k: trees k=100
    :return: prints evaluation to terminal
    """
    def __init__(self, data, test_size=0.2, k=100):
        super().__init__(data, test_size)

        self.k = k
        self.sensitivity, self.specificity, self.predictions = int(), int(), None

        self.run_classifier()

    def run_classifier(self):
        # train the model
        self.model = self.train_model()

        # make predictions
        self.predictions = self.model.predict(self.x_test)
        # get evaluation
        # self.evaluate()

        # Print results
        print(r2_score(self.y_test, self.predictions))
        print(dict(zip(self.x_test.columns, self.model.feature_importances_)))

    def train_model(self):
        neigh = RandomForestRegressor(n_estimators=self.k)
        neigh.fit(self.evidence, self.labels)
        return neigh

    def evaluate(self):
        raise NotImplementedError

    def __str__(self):
        return "This method implements the random forest classification."

    def print_results(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
        # TODO


if __name__ == "__main__":
    main()
