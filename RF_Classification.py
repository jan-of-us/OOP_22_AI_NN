from data_import import import_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Classification import Classification


def main():
    # import test data
    data = import_csv("Data/divorce.csv")
    # Create classifier class
    classifier = RF_Classification(data)
    print(classifier)
    print(classifier.print_results())


class RF_Classification(Classification):
    """
    RandomForest-classification.
    :param evidence: array of evidence, int or float,
    :param labels: list of labels int
    :param test_size: Size of testing data 0-1, default: 0.2 float,
    :param k: trees k=100
    :return: prints evaluation to terminal
    """
    def __init__(self, data, test_size=0.2, n=100):
        super().__init__(data, test_size)

        self.n = n
        self.sensitivity, self.specificity, self.predictions = int(), int(), None

        self.run_classifier()

    def run_classifier(self):
        # train the model
        self.model = self.train_model()

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
        neigh = RandomForestClassifier(n_estimators=self.n)
        neigh.fit(self.evidence, self.labels)
        return neigh

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

    def plot(self):
        raise NotImplementedError
        # TODO


if __name__ == "__main__":
    main()
