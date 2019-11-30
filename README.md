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

Data exploration

Exploratory Data Analysis is a very important step before training the model. MEDV it is the target and we will plot the distribution of the target to understand the connection with the features.
![Matrix](./fIgures/price_distribution.png)

Next we are creating a correlation matrix that measures the linear relationships between the variables. The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that there is a strong positive correlation between the two variables. When it is close to -1, the variables have a strong negative correlation
![Matrix](./fIgures/correlation_matrix.png)

After exploring the data our research cannot use NOX (-0.43) or INDUS (-0.48) because both of them have high negative correlation with MEDV.
To fit a linear regression model, we select those features which have a high correlation with our target variable MEDV. 
An important point in selecting features for a linear regression model is to check for multi-co-linearity. 
Based on the analise of the correlation matrix we will use RM and LSTAT as our features. 


Methods

We are going to use a linear regression model with MEDV as the dependent variable and all the others variable as independent variables.
The data will be split into training and testing sets and we train the model with 40% of the data and test with the remaining 60%. 

To measure the model’s performance we will calculate the coefficient of determination R2.

R2 = 0.6882607142538019

![Prices vesus Predicted Prices](./fIgures/prices_pred_price.png)

Conclusion

Our project been looking to find what are the environmental factors that influence the price of houses in Boston but we cannot answer.
The created model could provide a prediction of prices but we did not have any feature that could predict how the prices will be affected by the global warming.


References

https://www.boston.gov/environment-and-energy/preparing-climate-change

https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155

https://scikit-learn.org
