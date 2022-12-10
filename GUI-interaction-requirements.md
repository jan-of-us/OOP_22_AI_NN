## GUI Requirements
### Inputs
* Classification
    * choosing the type of classification (not final):
        * KNeighbours
        * Machine learning / NN 
        * Random Forest
    * specifying various parameters:
        * train / test splits
        * for KNeighbours: K
        * for NN: training time, activation functions, hidden layer count and type (?)
    * tbd: specify input data format, split into evidence & labels?, numeric values, easy example: evidence: list of lists, labels: list

### Outputs
* classes will have options for outputting the data to the user:
    * print - text based outputs like accuracy, specificity
    * plot methods - plots, confusion matrices
    
### Class structure
* create class ex. KNeighbours(Data=x, K=1, split=0.8)
* Class has a method called run or similar for processing
* plot or print to output the results, various parameters to decide format / type etc
* Possible wrapper class that calls all the other classes, not sure yet if possible
* show dataframe method to show manipulated data