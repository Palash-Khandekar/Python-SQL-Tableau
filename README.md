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
```
# create the special class that we are going to use from here on to predict new data
class absenteeism_model():
       
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
         
        # take a data file (*.csv) and preprocess 
        def load_and_clean_data(self, data_file):
             
            # import the data
            df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            # drop the 'ID' column
            df = df.drop(['ID'], axis = 1)
            # to preserve the code we've created in our other file, we will add a column with 'NaN' strings
            df['Absenteeism Time in Hours'] = 'NaN'
 
            # create a separate dataframe, containing dummy values for ALL avaiable reasons
            reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
             
            # split reason_columns into 4 types
            reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
            reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
            reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
            reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
             
            # to avoid multicollinearity, drop the 'Reason for Absence' column from df
            df = df.drop(['Reason for Absence'], axis = 1)
            #drop service time as we do not have much information about it
            df = df.drop(['Service time'], axis = 1)
            #drop seasons as we will use day of the week and month
            df = df.drop(['Seasons'], axis = 1)
             
            # concatenate df and the 4 types of reason for absence
            df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
             
            # assign names to the 4 reason type columns
            column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                           'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children','Social smoker', 
                           'Social drinker','Pet', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
            df.columns = column_names
 
            # re-order the columns in df
            column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 
                                      'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 
                                      'Children', 'Pet','Social smoker','Social drinker' 'Absenteeism Time in Hours']
            df = df[column_names_reordered]
       
            # convert the 'Date' column into datetime
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
 
            # create a list with month values retrieved from the 'Date' column
            list_months = []
            for i in range(df.shape[0]):
                list_months.append(df['Date'][i].month)
 
            # insert the values in a new column in df, called 'Month Value'
            df['Month Value'] = list_months
 
            # create a new feature called 'Day of the Week'
            df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
 
 
            # drop the 'Date' column from df
            df = df.drop(['Date'], axis = 1)
 
            # re-order the columns in df
            column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                                'Transportation Expense', 'Distance to Work', 'Age',
                                'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                                'Pet','Social smoker','Social drinker' 'Absenteeism Time in Hours']
            df = df[column_names_upd]
 
 
            # map 'Education' variables; the result is a dummy
            df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
 
            # replace the NaN values
            df = df.fillna(value=0)
 
            # drop the original absenteeism time
            df = df.drop(['Absenteeism Time in Hours'],axis=1)
             
            # drop the variables we decide we don't need
            df = df.drop(['Day of the Week','Daily Work Load Average','Distance to Work','Education'],axis=1)
             
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
             
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
     
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
         
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
         
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data
```


## Result
* The accuracy of the training model is about 78.39%.
* The accuracy of the model when given a set of new data is 73.57%.


