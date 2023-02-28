from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Classification_Data:
    # INPUTS
    # data
    data: pd.dataframe
    # Share/Percentage of Data used for testing
    test_size: float = 0.2
    # with x_labels, if None all will be used
    x_labels: str = None
    # title of column with labels for classification, if None last column will be used
    y_label: str = None
    # number and nodes for hidden layers as array 3 layers with 64 nodes each: [64, 64, 64]
    hidden_layers: list[int] = (64, 64)
    # number of training epochs
    training_epochs: int = 10
    # activation functions from tf.keras.activations
    activation_func: str = "relu"
    # Whether during the training a part of the data will already be used for testing after each epoch,
    # needed for accuracy/loss per epoch graphs
    validation_split: bool = True
    # Number of trees
    trees: int = 100

    # OUTPUTS
    # Plots
    confusion_matrix = None
    accuracy_per_epoch = None
    feature_importance = None

    # text based
    result_string = None


