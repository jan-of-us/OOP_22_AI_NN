from data_import import import_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    # import test data TODO get rid of hard coding (?)
    evidence, labels = import_csv("Data/divorce.csv")
    # Create classifier class
    classifier = KN_Classification(evidence, labels)
    # run classification
    #classifier.run_classifier() TODO: Discuss if run should be part of init or not
    # plot results TODO


class KN_Classification:
    """
    K-Neighbours-classification. Default k=1
    :param evidence -> array of evidence, int or float,
    :param labels -> list of labels int
    :param test_size -> Size of testing data 0-1, default: 0.2 float,
    :param k -> neighbours to be considered for classification int
    :return: tbd, prints evaluation to terminal
    """
    def __init__(self, evidence, labels, test_size=0.2, k=1):
        self.evidence = evidence
        self.labels = labels
        self.test_size = test_size
        self.k = k

        # split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.evidence, self.labels, test_size=self.test_size
        )
        self.run_classifier()

    def run_classifier(self):
        # train the model
        model = self.train_model(self.x_train, self.y_train, self.k)
        # make predictions
        predictions = model.predict(self.x_test)
        # get evaluation
        sensitivity, specificity = self.evaluate(self.y_test, predictions)

        # Print results
        print(f"Correct: {(self.y_test == predictions).sum()}")
        print(f"Incorrect: {(self.y_test != predictions).sum()}")
        print(f"True Positive Rate: {100 * sensitivity:.2f}%")
        print(f"True Negative Rate: {100 * specificity:.2f}%")

    def train_model(self, evidence, labels, n=1):
        neigh = KNeighborsClassifier(n_neighbors=n)
        neigh.fit(evidence, labels)
        return neigh

    def evaluate(self, labels, prediction) -> tuple:
        pos_label = 0
        neg_label = 0
        corr_pos_pred = 0
        corr_neg_pred = 0
        for i in range(len(labels)):
            # specificity
            if labels[i] == 0:
                neg_label += 1
                if prediction[i] == 0:
                    corr_neg_pred += 1
            # sensitivity
            if labels[i] == 1:
                pos_label += 1
                if prediction[i] == 1:
                    corr_pos_pred += 1
        sensitivity = corr_pos_pred / float(pos_label)
        specificity = corr_neg_pred / float(neg_label)
        return sensitivity, specificity

    def plot(self):
        ...
        # TODO


if __name__ == "__main__":
    main()
