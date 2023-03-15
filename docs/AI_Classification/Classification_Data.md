#


## Classification_Data
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/Classification_Data.py\#L7)
```python 
Classification_Data()
```
## Parameters


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



