# Project 2022 - AI and Neural Networks

## Implementation into the GUI
### User Inputs:
Classification_Data and Regression_Data implement all user inputs and handle outputs from the classes.

| Input Name          | Type             | Range / Values                      | Default Value  | Used for       | Description                                                                                                                                 | Implementation suggestions |
|---------------------|------------------|-------------------------------------|----------------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------|----------------------------|
| Data                | Pandas Dataframe | -                                   | -              | General        | Contains the dataset                                                                                                                        ||
| test_size           | float            | 0.2-0.8                             | 0.2            | General        | Share/Percentage of Data used for testing, if pretrained model is used, all data (0.99) will be used for testing                            ||
| x_labels            | list[str]        | headers from dataframe              | None           | General        | Labels used as evidence for the classification, if None all but y will be used                                                              |                            |
| y_label             | str              | header from dataframe               | None           | General        | Label of column that contains the classes, if None final column will be used                                                                |                            |
| hidden_layers       | array of ints    | [32]-[4096, 4096, 4096, 4096, 4096] | [64, 64]       | Neural Net     | Nodes for each hidden layer, every entry in the array creates a hidden layer with as many nodes as the entry's value                        | textbox with csv           |
| training_epochs     | int              | 1 - 200                             | 10             | Neural Net     |                                                                                                                                             ||
| activation_func     | string           |                                     | "relu"         | Neural Net     | https://www.tensorflow.org/api_docs/python/tf/keras/activations                                                                             | dropdown menu              |
| validation_split    | Bool             |                                     | True           | Neural Net     | Whether during the training a part of the data will already be used for testing after each epoch, needed for accuracy/loss per epoch graphs | checkbox                   |
| trees               | int              | 1 - 10.000                          | 100            | Random Forest  | Number of trees in the forest                                                                                                               ||
| model               |                  |                                     | None           | General        | Allows user uploaded pre-trained models                                                                                                     ||
|                     |                  |                                     |                |                |                                                                                                                                             |




### Outputs:
#### General:
  * result_string contains results in text based form
  * feature_importance_dict: dictionary with the importance of each attribute
  * accuracy_score
#### Plots
* confusion matrix
* for NN: accuracy for each epoch training + testing, losses
* feature importance pie chart

### Example Code to run the neural network classifier and plot the confusion matrix:
```
pip install -r requirements.txt
```

```
from NN_Classification import NN_Classification
from Classification_Data import Classification_Data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

data = pd.read_csv(filename, sep=";")
data_obj = Classification_Data(data=data)
classifier = NN_Classification(data_obj)
plt.show()
```


<details>

### Old: Usage Prototype 1 KNeighbours
#### data_import.py
used to import .csv data (only numeric values) into a combination of list of lists for evidence and a list for the labels
input: filepath
returns: evidence, labels

Ex.: to import divorce.csv create folder Data in root dir of repository and put divorce.csv inside. Call import_csv('Filepath')
</details>


### Kickoff meeting
<details>

#### Prerequisites

* github/lab or similar for version control
* Choice of a documentation system, example Mkdocs
* choices in GUI:
  * Classification
    * Neural Networks
    * Random forest
    * discrete vs continuous?
  * Regression?
  * choices of different variables, epochs, layer types, constellations etc?
  * splits, training length
* inputs:
  * training and testing data as split inputs?
  * 
* data must be in list/array etc
* all values must be int/float, no strings etc -> gui/architecture group or our own work?
* Outputs:
  * Classification stats (accuracy for training and testing) with plots
  * confusion matrices
  * mean and standard deviation
  * data info: class distributions, ...
  * influence of each input variable and the result?
</details>

### Roadmap
<details>

#### Prototype 1 - classification only

* read data
* load data into list/array
* classify data using the KNeighbours algorithm
* show basic info, accuracy

#### Prototype 2

* Neural Network classifier

#### Next steps current Prototype
* further implementation of super class Classification for previous prototypes
* implementing all needed methods
* adding more plotting and display options
* implementing more user influences ex. neural network layers, activation functions etc.
 
#### To-Do Afterwards
* revising the KNeighbours Classification
* Implementing the random forest classifier
* Adding a method for the timeseries dataset
* Testing on different datasets

</details>

### Links
<details>

* Git Crash Course: https://www.youtube.com/watch?v=SWYqp7iY_Tc
* Tensorflow documentation: https://www.tensorflow.org/
* TF RandomForest: https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel
* TF Regression: https://www.tensorflow.org/tutorials/keras/regression
* TF Classification example: https://www.tensorflow.org/tutorials/keras/classification
* MkDocs (example for documentation framework): https://www.mkdocs.org/
* Git cheat sheet: https://education.github.com/git-cheat-sheet-education.pdf
* Lecture on Neural Networks: https://cs50.harvard.edu/ai/2020/notes/5/
</details>

#### Other
* ...

#### Questions

* ...
