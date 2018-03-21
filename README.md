# Linear_Regression_Assignment
Assignment for use opendata to predict PM2.5 with Linear Regression

## Problem Description
Use continuous twelve hours data with opendata of air to predict PM2.5 values, you must implement it by linear regression.

## Data Description
- train.csv: opendata which had recorded 12 months data of air.Each hours record 18 attributes, and each months recorded 20 days for training set, remained day for testing set.
![avatar](https://i.imgur.com/Lz72PxL.png)
- sample-test.csv: sapmle testing set to let you test how accurancy your model is.
- predictions.csv: use model to predict sample-test.csv data.
- sample-test-outputs.csv: correct answer of sample-test.csv.

## Use Language & Packages
- Python 
- Numpy 
- Pandas

## Start Training
In command Line, use command
```console
python  Linear_Regression.py --mode train --file [filename]
```
to start training 'train.csv',note that training only work with 'train.csv', if you want to use to other train-data, you must modify the code.
![avatar](https://i.imgur.com/Sx37Jdm.png)
after finish training,it will output a file 'data.model', which save the linear regression model.

## Start Testing
In command Line, use command
```console
python  Linear_Regression.py --mode test --file [filename]
```
![avatar](https://i.imgur.com/neSXSaj.png)
to start testing, the data size of testing-set can be different, it will auto-compute the data size.

## Output Predict Result
after finish testing, it will output a file which name is 'predictions.csv' that recorded the predict PM2.5 value.
![atavar](https://i.imgur.com/dt84eMT.png)
