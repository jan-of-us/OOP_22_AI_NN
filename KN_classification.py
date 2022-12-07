from data_import import import_csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


TEST_SIZE = 0.2

def main():
    # import test data TODO get rid of hard coding
    evidence, labels = import_csv("Data/divorce.csv")
    # split into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    # train the model
    model = train_model(x_train, y_train)
    # make predictions
    predictions = model.predict(x_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def train_model(evidence, labels, n=1):
    neigh = KNeighborsClassifier(n_neighbors=n)
    neigh.fit(evidence, labels)
    return neigh


def evaluate(labels, prediction):
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


if __name__ == "__main__":
    main()