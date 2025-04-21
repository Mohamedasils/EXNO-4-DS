# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
NAME : MOHAMED ASIL S
REGISTER NUMBER : 212223040112
```

```
import pandas as pd
import numpy as np
```
```
df=pd.read_csv('/content/bmi.csv')
df
```
IMG
```
df.head()
```
IMG
```
df.dropna()
```
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
199
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
IMG
```
df1=pd.read_csv('/content/bmi.csv')
df2=pd.read_csv('/content/bmi.csv')
df3=pd.read_csv('/content/bmi.csv')
df4=pd.read_csv('/content/bmi.csv')
df5=pd.read_csv('/content/bmi.csv')
df5
```
IMG
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df.head(10)
```
IMG
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
IMG
```
from sklearn.preprocessing import MaxAbsScaler
max1=MaxAbsScaler()
df3[['Height','weight']]=max1.fit_transform(df3[['Height','Weight']])
df3
```
IMG
```
from sklearn.preprocessing import RobustScaler
roub=RobustScaler()
df4[['Height','Weight']]=roub.fit_transform(df4[['Height','Weight']])
df4
```
IMG
```
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_classif
from sklearn.feature_selection import chi2
data=pd.read_csv('/content/income.csv')
data
```
IMG
```
data1=pd.read_csv('/content/titanic_dataset (1).csv')
data1
```
IMG

# RESULT:
       # INCLUDE YOUR RESULT HERE
