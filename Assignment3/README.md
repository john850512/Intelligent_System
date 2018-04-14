# Linear_Regression_Assignment
Assignment for use opendata to predict PM2.5 with Linear Regression

## Problem Description
Use 106 featutes(data had been preprocessed) to prdict a person is rich or poor(make over 50K yes or not).

## Data Description
- train_X: preprocessed-data, had finished one-hot encoding with discrete features, features with continuous remain the same.  
<img src="/img/img1.PNG" height="300" width="400" >

- train_Y: result for train_X, 1 means rich and 0 means poor.
- predictions.csv: use model to predict test_X data.
- others file(.model): training model. 

## Use Language & Packages
- Python 
- Numpy 

## Start Training
In this assignment, just command the `testing block` and uncommand `training block`, and compile the code to start training.
after finish training,it will output some file(.model) , which save the logistic regression model.

## Start Testing
just command the `training block` and uncommand `testing block`, and compile the code to start testing.
the data size of testing-set can be different, it will auto-compute the data size.

## Output Predict Result
after finish testing, it will output a file which name is 'predictions.csv' that recorded the prediction.
![atavar]()

## More Detail
I use different strategy to implement this assignment, below show three result to descripte the training result with each strategy.
- Pure Logistic Regression
- Standardization
- Standardization & Normalization
