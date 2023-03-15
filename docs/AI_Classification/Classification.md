#


## Classification
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/Classification.py\#L10)
```python 
Classification(
   data_obj: Classification_Data
)
```


---
Class for all classification methods


**Methods:**


### .encode
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/Classification.py\#L33)
```python
.encode()
```

---
Encodes variables that are not integer or float format


**Returns**

converted dataframe

### .split_evidence_labels
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/Classification.py\#L44)
```python
.split_evidence_labels(
   data_obj
)
```

---
Splits given dataset into evidence and labels


**Args**

* **data_obj**  : Classification_Data object


### .plot_confusion_matrix
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/Classification.py\#L72)
```python
.plot_confusion_matrix(
   y_test, predictions, title
)
```

---
Generates a confusion matrix with given labels and predictions


**Args**

* **y_test**  : real labels
* **predictions**  : predicted labels
* **title**  : Title for the plot


**Returns**

matplotlib subplot
