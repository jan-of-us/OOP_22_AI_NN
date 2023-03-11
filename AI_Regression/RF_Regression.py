import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from AI_Regression.Regression import Regression
from AI_Regression.Regression_Data import Regression_Data
import pickle


class RF_Regression(Regression):
    """
    RandomForest-classification.
    :param data_obj: Regression_Data object
    :return: data_obj with filled result variables
    """
    def __init__(self, data_obj: Regression_Data):
        super().__init__(data_obj)

        self.k = data_obj.trees
        self.sensitivity, self.specificity, self.predictions = int(), int(), None
        self.run_classifier(data_obj)

        self.plot(data_obj)

    def run_classifier(self, data_obj):
        # train the model
        if data_obj.model is not None and isinstance(data_obj.model, RandomForestRegressor):
            self.model = data_obj.model
            print("Model loaded")
        else:
            self.model = self.train_model()
            data_obj.model = self.model
            print("Model created")

        # make predictions
        self.predictions = self.model.predict(self.x_test)
        # get evaluation
        self.evaluate(data_obj)


        # Print results
        self.print_results(data_obj)


    def train_model(self):
        forest = RandomForestRegressor(n_estimators=self.k)
        forest.fit(self.x_train, self.y_train)
        return forest

    def evaluate(self, data_obj):
        data_obj.r2_score = r2_score(self.y_test, self.predictions)
        data_obj.mean_abs_error = mean_absolute_error(self.y_test, self.predictions)
        data_obj.mean_sqr_error = mean_squared_error(self.y_test, self.predictions)
        data_obj.feature_importance_dict = dict(zip(self.x_test.columns, self.model.feature_importances_))

    def __str__(self):
        return "This method implements the random forest regression."

    def print_results(self, data_obj):
        data_obj.result_string += f"The regressors R2_Score is {data_obj.r2_score}.\n\n" \
                                 f"The mean abs. error is {data_obj.mean_abs_error}.\n\n" \
                                 f"The mean squared error is {data_obj.mean_sqr_error}."

    def plot(self, data_obj):
        # plot predictions
        super().plot_predictions(y_scaler=self.y_scaler, y_test=self.y_test, predictions=self.predictions, data_obj=data_obj)

        # feature importance pie chart
        fig, ax = plt.subplots()
        ax.pie(data_obj.feature_importance_dict.values(), labels=data_obj.feature_importance_dict.keys())
        ax.axis('equal')
        data_obj.feature_importance = fig


def main():
    # import test data
    data = pd.read_csv("../Data/energydata_complete.csv")

    # Create Data Class, start index and n_values atm only used for plotting, training and prediction done on all data
    data_obj = Regression_Data(data=data, x_labels=["T1"], y_label="Appliances", n_values=50, test_size=0.2, trees=100, scale=False)

    filename = '../model.sav'
    # data_obj.model = pickle.load(open(filename, 'rb'))

    # Create classifier class
    regressor = RF_Regression(data_obj)
    # pickle.dump(data_obj.model, open(filename, 'wb'))
    plt.show()
    print(data_obj.result_string)



if __name__ == "__main__":
    main()
