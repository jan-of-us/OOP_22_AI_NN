import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from Regression import Regression
from Regression_Data import Regression_Data
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


    # Create Data Class
    data_obj = Regression_Data(data=data)
    print(data_obj)
    # Create classifier class
    regressor = RF_Regression(data_obj)
    plt.show()


class RF_Regression(Regression):
    """
    RandomForest-classification.
    :param evidence: array of evidence, int or float,
    :param labels: list of labels int
    :param test_size: Size of testing data 0-1, default: 0.2 float,
    :param k: trees k=100
    :return: prints evaluation to terminal
    """
    def __init__(self, data_obj: Regression_Data):
        super().__init__(data_obj.data, data_obj.test_size)

        self.k = data_obj.trees
        self.sensitivity, self.specificity, self.predictions = int(), int(), None
        self.run_classifier(data_obj)

        self.plot(data_obj)

    def run_classifier(self, data_obj):
        # train the model
        self.model = self.train_model()

        # make predictions
        self.predictions = self.model.predict(self.x_test)
        # get evaluation
        # self.evaluate()

        data_obj.r2_score = r2_score(self.y_test, self.predictions)
        data_obj.mean_abs_error = mean_absolute_error(self.y_test, self.predictions)
        data_obj.mean_sqr_error = mean_squared_error(self.y_test, self.predictions)
        data_obj.feature_importance_dict = dict(zip(self.x_test.columns, self.model.feature_importances_))

        # Print results

        #print(self.feature_importance)

    def train_model(self):
        neigh = RandomForestRegressor(n_estimators=self.k)
        neigh.fit(self.evidence, self.labels)
        return neigh

    def evaluate(self):
        raise NotImplementedError

    def __str__(self):
        return "This method implements the random forest classification."

    def print_results(self):
        # TODO, part of data class??
        raise NotImplementedError

    def plot(self, data_obj):
        fig, ax = plt.subplots()
        plt.plot(self.y_test["lights"].to_numpy()[:50], color='red', label='Real data')
        plt.plot(self.predictions[:, 1][:50], color='blue', label='Predicted data')
        plt.title('Prediction')
        plt.legend()
        data_obj.prediction = fig

        # feature importance pie chart
        fig, ax = plt.subplots()
        ax.pie(data_obj.feature_importance_dict.values(), labels=data_obj.feature_importance_dict.keys())
        ax.axis('equal')
        data_obj.feature_importance = fig



if __name__ == "__main__":
    main()
