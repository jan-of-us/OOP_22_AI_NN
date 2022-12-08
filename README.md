# Project 2022

## AI and Neural Networks

### Usage Prototype 1 KNeighbours
#### data_import.py
used to import .csv data (only numeric values) into a combination of list of lists for evidence and a list for the labels
input: filepath
returns: evidence, labels

Ex.: to import divorce.csv create folder Data in root dir of repository and put divorce.csv inside. Call import_csv('Filepath')



### Kickoff meeting

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

#### Prototype 1 - classification only

* read data
* load data into list/array
* train a simple NN
* test on that NN
* show basic info, accuracy
* KNeighbours classifier for now

#### Prototype 2

* ...

#### Preparation

* Tensorflow documentation: https://www.tensorflow.org/
* TF RandomForest: https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel
* TF Regression: https://www.tensorflow.org/tutorials/keras/regression
* TF Classification example: https://www.tensorflow.org/tutorials/keras/classification
* MkDocs (example for documentation framework): https://www.mkdocs.org/
* Git cheat sheet: https://education.github.com/git-cheat-sheet-education.pdf
* Lecture on Neural Networks: https://cs50.harvard.edu/ai/2020/notes/5/

#### Other

* github names to create repository
  * ...

#### Questions

* ?