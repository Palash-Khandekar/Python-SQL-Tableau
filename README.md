# Prediction of Absenteeism at Work using Python and SQL.
The aim of this project is to understand the flow of data in a real life business model. To understand this, I have integrated Jupyter notebook, MySQL workbench and tableau. The figure below shows the responsibilities of Data Scientists/ML Engineers and Business Intelligence/Data Analysts.

![sql-jupy-tab](./Images/tab-sql-jupy.PNG)

## Table of contents
* [Business Understanding](#business-understanding)
* [Technologies and Tools](#technologies-and-tools)
* [Flow of Process](#flow-diagram)
* [Important Libraries](#lib)
* [Code Example](#code-example)
* [Result](#result)

## Business Understanding
Absenteeism is the term given when an employee is habitually and frequently absent from work. This excludes paid leave and occasions where an employer has granted an employee time off. According to [Forbes](https://www.forbes.com/sites/investopedia/2013/07/10/the-causes-and-costs-of-absenteeism-in-the-workplace/#4af53573eb65), Absenteeism costs U.S. companies billions of dollars each year in lost productivity, wages, poor quality of goods/services and excess management time. In addition, the employees who do show up to work are often burdened with extra duties and responsibilities to fill in for absent employees, which can lead to feelings of frustration and a decline in morale.It is important for a company to understand the causes of absenteeism and make policies inorder to reduce these causes. 
In this project, I will build a machine learning model to predict the absenteeism. 

The data for this analysis is taken from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work#)

## Technologies and Tools:
* Python(Jupyter Notebook)
* MySQL Workbench
* Tableau

## Steps:
* Data Preprocessing
* Machine Learning model - Logistic Regression
* Connect MySQL workbench and Jupyter Notebook
* Create database and table in MySQL
* Insert data in MySQL from Jupyter Notebook
* Save the predicted_output from MySQL as a csv file
* Analyze the results in Tableau

## Important Libraries
* Connection between MySQL workbench and Jupyter Notebook
![sql-jupy](./Images/mysql-jupy.PNG)

## Code Example


## Result
* The accuracy of the training model is about 79.28%.
* The accuracy of the model when given a set of new data is 74.28%.


