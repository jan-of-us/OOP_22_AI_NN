#


## Regression
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/Regression.py\#L9)
```python 
Regression(
   data_obj: Regression_Data
)
```


---
Class for all regression methods


**Methods:**


### .process_data
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/Regression.py\#L43)
```python
.process_data(
   data_obj
)
```

---
Drop non-numeric columns from the dataframe, tries to find a column with date and converts it to usable format


**Args**

* **data_obj**  : Regression_Data object


**Returns**

modified self.data and data_obj.result_string to inform user
of processing

### .split_data
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/Regression.py\#L101)
```python
.split_data(
   data_obj
)
```

---
Splits given dataset into evidence and labels, requires labels to be last column of dataframe


**Args**

* **data_obj**  : Regression_Data object


**Returns**

modified data_obj

### .print_results
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/Regression.py\#L131)
```python
.print_results(
   data_obj
)
```

---
Adds the results to the result_string for the GUI


**Args**

* **data_obj**  : Regression_Data object


**Returns**

modified data_obj

### .plot_predictions
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/Regression.py\#L145)
```python
.plot_predictions(
   y_scaler, y_test, predictions, data_obj, train_test
)
```

---
Plots the predicted and real values against each other. Plots as many values as the user selected


**Args**

* **y_scaler**  : MinMaxScaler used to scale y-data
* **y_test**  : Real values for y
* **predictions**  : Predicted values for y
* **data_obj**  : Regression_Data object
* **train_test**  : str "train" or "test" for plot title and correct output to data_obj


**Returns**

modified data_obj
