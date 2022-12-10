import tensorflow as tf
from data_import import import_csv
from Classification import Classification
from sklearn.model_selection import train_test_split
import numpy as np

class NN_Classification(Classification):
    def __init__(self, data, test_size=0.2, epochs=10):
        super().__init__(data, test_size=0.2)
        self.epochs = epochs

        # TEMPORARY TODO Implement super methods based on GUI Dataformat
        #self.evidence = evidence
        #self.labels = labels
        #self.labels = tf.keras.utils.to_categorical(self.labels)
        #self.test_size = test_size
        #self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        #    np.array(self.evidence), np.array(self.labels), test_size=self.test_size
        #)


        self.get_model()
        print(self.x_train)
        print(self.y_train)
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs)
        self.model.evaluate(self.x_test, self.y_test, verbose=2)

    def get_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(len(self.x_train[0]))),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


def main(file):
    data = import_csv(file)
    classifier = NN_Classification(data)


if __name__ == "__main__":
    main("Data/divorce.csv")
