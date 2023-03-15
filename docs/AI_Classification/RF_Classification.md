#


## RF_Classification
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/RF_Classification.py\#L10)
```python 
RF_Classification(
   data_obj: Classification_Data
)
```




**Methods:**


### .run_classifier
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/RF_Classification.py\#L25)
```python
.run_classifier(
   data_obj
)
```

---
Initialize the model, train (if not loaded) and evaluate on test data


**Args**

* **data_obj**  : Classification_Data object


**Returns**

data_obj with modified variables

### .train_model
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/RF_Classification.py\#L65)
```python
.train_model()
```

---
Initializes and trains the random forest


**Returns**

RandomForestClassifier

### .plot
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Classification/RF_Classification.py\#L80)
```python
.plot(
   data_obj
)
```

---
Creates the plots


**Args**

* **data_obj**  : Classification_Data object


**Returns**

data_object with modified variables
