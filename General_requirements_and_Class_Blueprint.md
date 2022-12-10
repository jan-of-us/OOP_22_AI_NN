## General requirements
* Input Data Format: np.ndarray, pd.DataFrame, something else ? TDB!
* return Data must have same format
* Plots: matplotlib.pyplot or plotly -> final format TBD
* If you use any special modules that require specific python versions, please specify, ex. Tensorflow only works with Python 3.9 atm
* All groups should have a requirements.txt document that lists needed packages for their program, it can be installed with pip install -r requirements.txt
* All groups need to have a file (md?) describing how the module gets called in the gui, which input values you need, which types of graphs can be displayed etc
* 

## Class structure
```
class Class_Name_Here:
    def __init__(self):
        Class should run after initialization. 
        
    Additional methods the class must have:
    
    def return_data(self) -> InputDataFormat:
        returns the manipulated data

    def __str__(self) -> str:
        returns a string with whatever info you want to display to the user

    def plot(self, parameter_for_plot_selection) -> PlotFormat:
        with the parameter output different type of plots, if you only need one it's not needed

```