# Project 2022 - AI and Neural Networks


Created as part of [Data Science and Visualization Application](https://github.com/YuganshuWadhwa/Data-Science-and-Visualization-Application).

For example usage see [Implementation Examples](Implementation_Examples.ipynb)
## Implementation into the GUI
### User Inputs:
Classification_Data and Regression_Data implement all user inputs and handle outputs from the classes.
### Classification Data Parameters:
<details>

| Parameter Name   | Type             | Range / Values                                                                                              | Default Value  | Used for       | Description                                                                                                                                 |
|------------------|------------------|-------------------------------------------------------------------------------------------------------------|----------------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Data             | Pandas Dataframe | -                                                                                                           | -              | General        | Contains the dataset                                                                                                                        |
| test_size        | float            | 0.2-0.8                                                                                                     | 0.2            | General        | Share/Percentage of Data used for testing, if pretrained model is used, all data (0.99) will be used for testing                            |
| x_labels         | list[str]        | headers from dataframe                                                                                      | None           | General        | Labels used as evidence for the classification, if None all but y will be used                                                              |
| y_label          | str              | header from dataframe                                                                                       | None           | General        | Label of column that contains the classes, if None final column will be used                                                                |
| hidden_layers    | array of ints    | [32]-[4096, 4096, 4096, 4096, 4096]                                                                         | [64, 64]       | Neural Net     | Nodes for each hidden layer, every entry in the array creates a hidden layer with as many nodes as the entry's value                        |
| training_epochs  | int              | 1 - 200                                                                                                     | 10             | Neural Net     |                                                                                                                                             |
| activation_func  | string           | elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, tanh, exponential, gelu, selu, softsign, swish | "relu"         | Neural Net     | https://www.tensorflow.org/api_docs/python/tf/keras/activations                                                                             |
| validation_split | Bool             |                                                                                                             | True           | Neural Net     | Whether during the training a part of the data will already be used for testing after each epoch, needed for accuracy/loss per epoch graphs |
| trees            | int              | 1 - 10.000                                                                                                  | 100            | Random Forest  | Number of trees in the forest                                                                                                               |
| model            |                  |                                                                                                             | None           | General        | Allows user uploaded pre-trained models                                                                                                     |

</details>

### Regression Data Parameters

<details>

| Input Name        | Type             | Range / Values                                                                                              | Default Value | Used for      | Description                                                                                                                                 |
|-------------------|------------------|-------------------------------------------------------------------------------------------------------------|---------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Data              | Pandas Dataframe | -                                                                                                           | -             | General       | Contains the dataset                                                                                                                        |
| test_size         | float            | 0.2-0.8                                                                                                     | 0.2           | General       | Share/Percentage of Data used for testing, if pretrained model is used, all data (0.99) will be used for testing                            |
| x_labels          | list[str]        | headers from dataframe                                                                                      | None          | General       | Labels used as evidence for the classification, if None all but y will be used                                                              |
| y_label           | str              | header from dataframe                                                                                       | None          | General       | Label of column that contains the classes, if None final column will be used                                                                |
| n_values          | int              | 20 - test_size*len(data)                                                                                    | 50            | General       | Determines how many values are plotted in output graphs                                                                                     |
| hidden_layers     | array of ints    | [32]-[4096, 4096, 4096, 4096, 4096]                                                                         | [64, 64]      | Neural Net    | Nodes for each hidden layer, every entry in the array creates a hidden layer with as many nodes as the entry's value                        |
| training_epochs   | int              | 1 - 200                                                                                                     | 10            | Neural Net    |                                                                                                                                             |
| activation_func   | string           | elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, tanh, exponential, gelu, selu, softsign, swish | "relu"        | Neural Net    | https://www.tensorflow.org/api_docs/python/tf/keras/activations                                                                             |
| trees             | int              | 1 - 10.000                                                                                                  | 100           | Random Forest | Number of trees in the forest                                                                                                               |
| model             |                  |                                                                                                             | None          | General       | Allows user uploaded pre-trained models                                                                                                     |

</details>

### Outputs:
#### General:
  * result_string contains results in text based form
  * feature_importance_dict: dictionary with the importance of each attribute
  * accuracy_score
#### Plots
* confusion matrix
* for NN: accuracy for each epoch training + testing, losses
* feature importance pie chart

### To install the needed modules:
```
pip install -r requirements.txt
```
