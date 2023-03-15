#


## RF_Regression
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/RF_Regression.py\#L10)
```python 
RF_Regression(
   data_obj: Regression_Data
)
```


---
RandomForest Regression.


**Args**

* **data_obj**  : Regression_Data object


**Returns**

data_obj with filled result variables


**Methods:**


### .run_regressor
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/RF_Regression.py\#L27)
```python
.run_regressor(
   data_obj
)
```

---
Load or create a model, train model (if applicable), make predictions for trained model or uploaded model if
it matches the data. Evaluate and plot results

**Args**

* **data_obj**  : Regression_Data object


**Returns**

data_obj with modified values

### .train_model
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/RF_Regression.py\#L68)
```python
.train_model()
```

---
Initialize the random forest


**Returns**

sklearn RandomForestRegressor

### .evaluate
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/RF_Regression.py\#L79)
```python
.evaluate(
   data_obj
)
```

---
Create the evaluation, R2 Score, MAE and MSE

**Args**

* **data_obj**  : Regression_Data object


**Returns**

data_obj with modified values

### .plot
[source](https://github.com/jan-of-us/OOP_22_AI_NN\blob\main\AI_Regression/RF_Regression.py\#L96)
```python
.plot(
   data_obj
)
```

---
Creates the output plots

**Args**

* **data_obj**  : Regression_Data object


**Returns**

data_obj with modified values
