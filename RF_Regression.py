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
    # Create classifier class
    regressor = RF_Regression(data_obj)
    regressor.plot()
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

        self.run_classifier()

    def run_classifier(self):
        # train the model
        self.model = self.train_model()

        # make predictions
        self.predictions = self.model.predict(self.x_test)
        # get evaluation
        # self.evaluate()

        self.r2 = r2_score(self.y_test, self.predictions)
        self.mas = mean_absolute_error(self.y_test, self.predictions)
        self.mse = mean_squared_error(self.y_test, self.predictions)
        self.feature_importance = dict(zip(self.x_test.columns, self.model.feature_importances_))

        # Print results
        print(self.r2, self.mse, self.mas)
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
        print(self.r2)
        print(self.feature_importance)

    def plot(self):
        fig, ax = plt.subplots()
        plt.plot(self.y_test["lights"].to_numpy()[:50], color='red', label='Real data')
        plt.plot(self.predictions[:, 1][:50], color='blue', label='Predicted data')
        plt.title('Prediction')
        plt.legend()
        return fig

        # feature importance pie chart
        #fig, ax = plt.subplots()
        #ax.pie(self.feature_importance.values(), labels=self.feature_importance.keys())
        #ax.axis('equal')




if __name__ == "__main__":
    main()
