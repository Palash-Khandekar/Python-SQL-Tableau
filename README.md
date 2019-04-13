# Prediction of Absenteeism at Work using Python and SQL.
The aim of this project is to understand the flow of data in a real life business model. To understand this, I have integrated Jupyter notebook, MySQL workbench and tableau. The figure below shows the responsibilities of Data Scientists/ML Engineers and Business Intelligence/Data Analysts.

![sql-jupy-tab](./Images/tab-sql-jupy.PNG)

## Table of contents
* [Problem Definition](#problem-definition)
* [Technologies and Tools](#technologies-and-tools)
* [Important Libraries](#important-libraries)
* [Code Example](#code-example)
* [Result](#result)

## Problem Definition
Absenteeism is the term given when an employee is habitually and frequently absent from work. This excludes paid leave and occasions where an employer has granted an employee time off. According to [Forbes](https://www.forbes.com/sites/investopedia/2013/07/10/the-causes-and-costs-of-absenteeism-in-the-workplace/#4af53573eb65), Absenteeism costs U.S. companies billions of dollars each year in lost productivity, wages, poor quality of goods/services and excess management time. In addition, the employees who do show up to work are often burdened with extra duties and responsibilities to fill in for absent employees, which can lead to feelings of frustration and a decline in morale.It is important for a company to understand the causes of absenteeism and make policies inorder to reduce these causes. <br>
**In this project, I will build a machine learning model to predict the absenteeism. The goal is to predict whether or not an employee presenting certain characteristics can be expected to be missing on a certain workday. Having such information in advance can help Managers in decision making by reorganizing the work process in such a way that will allow an organization to avoid lack of productivity and increase the quality of work.**

The data for this analysis is taken from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work#)

## Technologies and Tools:
* Python(Jupyter Notebook)
* MySQL Workbench
* Tableau

## Important Libraries
* Connection between MySQL workbench and Jupyter Notebook
![sql-jupy](./Images/mysql-jupy.PNG)

## Code Example
```
#import all libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

```
```
#customScaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns, copy = True, with_mean = True, with_std = True):
        self.scaler = StandardScaler(copy, with_mean, with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y = None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y = None, copy = None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
        X_notScaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_notScaled, X_scaled], axis = 1)[init_col_order]
```


## Result
* The accuracy of the training model is about 78.39%.
* The accuracy of the model when given a set of new data is 73.57%.


