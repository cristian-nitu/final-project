# final-project

Research Question

What are the environmental factors that influence the price of houses in suburbs of Boston, Massachusetts?

Introduction

Boston Dataset has been created based on the information collected by the U.S. Census Service concerning housing in the area of Boston Massachusetts.
The Boston Housing Dataset consists of price of houses in various places in Boston and the median value of house price in provided by MEDV ($1000) and we have a total of 14 attributes.
Alongside with price, the Boston dataset also provide information such as Crime (CRIM), portion of residential land zoned for lots over 25000 SQ/FT (ZN), areas of non-retail business in the town (INDUS), the age of people who own the house (AGE), the nitric oxides concentration (NOX), Charles River dummy variable (CHAS), average number of rooms per residence (RM), the walking distance to employments centers (DIS) ,index of accessibility to highways(RAD) ,full value of property-tax per $10000 (TAX), pupil-teacher ratio per town (PT)  and the percent of lower status of population(LSTAT) 

Challenge

The big challenge of Boston area is creating neighborhood solutions to coastal flooding from sea level rise and storms
City of Boston is preparing for climate change by creating The Vision Plan-first step in a long-term effort to increase recreational opportunities and respond to coastal flooding risks.

Objective

This report is looking to determine the influence of several neighborhood attributes on the prices of Boston housing. The specific dataset attributes to be considered are proximity to the Charles River and air pollution using nitrogen oxide concentrations as an explanatory variable for the median value of houses in Boston.
Our research is to create a working model that will have the capability to predict the house’s price using the above mentioned features: NOX and CHAS.

Methods

We are going to use a linear regression model with MEDV as the dependent variable and all the others variable as independent variables.
To measure the model’s performance will calculate the coefficient of determination R2.Any value between 0 and 1 indicate what percentage of the target variable can be explain by the features when we are using this model.

R2 score : 0.6882607142538019

![Matrix](./figures/prices_pred_price.png)


