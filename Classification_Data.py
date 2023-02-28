class Classification_Data:
    def __init__(self, data, test_size=0.2, x_labels=None, y_label=None, hidden_layers=[64, 64], training_epochs=10,
                 activation_func="relu", validation_split=True, trees=100):
        # pd.dataframe: data
        self.data = data
        # Float: Share/Percentage of Data used for testing
        self.test_size = test_size
        # list of strings: with x_labels, if None all will be used
        self.x_labels = x_labels
        # string: title of column with labels for classification, if None last column will be used
        self.y_label = y_label
        # Array: number and nodes for hidden layers as array 3 layers with 64 nodes each: [64, 64, 64]
        self.hidden_layers = hidden_layers
        # Int: number of training epochs
        self.training_epochs = training_epochs
        # String: activation functions from tf.keras.activations
        self.activation_func = activation_func
        # Bool: Whether during the training a part of the data will already be used for testing after each epoch,
        # needed for accuracy/loss per epoch graphs
        self.validation_split = validation_split
        # Int: Number of trees
        self.trees = trees


