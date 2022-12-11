import tensorflow as tf
from data_import import import_csv
from Classification import Classification


class NN_Classification(Classification):
    def __init__(self, data, test_size=0.2, epochs=10):
        super().__init__(data, test_size=0.2)
        self.y_train = tf.keras.utils.to_categorical(self.y_train)
        self.y_test = tf.keras.utils.to_categorical(self.y_test)

        self.epochs = epochs

        self.get_model()

        self.model.fit(self.x_train, self.y_train, epochs=self.epochs)
        self.model.evaluate(self.x_test, self.y_test, verbose=2)

    def get_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(self.evidence.shape[1])),
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
