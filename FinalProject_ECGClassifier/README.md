# ECGClassifier
Final Project of Intelligent_System, which goal is to classify five types of ECG data(one for normal, four for Arrhythmia).

## Problem Description
MIT-BIH dataset has ECG recorded and the annotation of data,which means the heart pluse is normal or not, we want to use machine learning technology to make
a classifier model for Arrhythmia.

## Data Description
We use MIT-BIH dataset as training / testing data, more info of MIT-BIH you can see here: https://www.physionet.org/physiobank/database/mitdb/  
<img src="./img/1.PNG">

In MIT-BIH, we only care about five annotation:
- N(Normal beat)
- L(Left bundle branch block beat)
- R(Right bundle branch block beat)
- a(Aberrated atrial premature beat)
- V(Premature ventricular contraction)

we will use tensorflow LSTM model to make a classifier those five annotation.

## Use Language & Packages
- Python 
- Numpy 
- Pandas
- WFDB

if you don't have packages listed above, use
``` console
pip install (package name)
```
to install it

## How to Start?
I wrote codes with jupyter-notebook, you can easily run my code

## Output Predict Result
after finish testing, the testing accurancy is about 95%
<img src="./img/2.PNG">


## More Detail
I make a PPT to declare the process of coding, In the future, I will also write a blog to describe the detail.
