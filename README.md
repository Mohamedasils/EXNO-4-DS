EXNO:4-DS
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

```python
import pandas as pd
from scipy import stats
import numpy as np
```

```python
df=pd.read_csv(r"D:\ds\EXNO-4-DS\bmi.csv")
df
```
![image](https://github.com/user-attachments/assets/d91c5503-1b13-4ced-a173-51963d844943)



```python
df.head()
```
![image-1](https://github.com/user-attachments/assets/bdaf3c22-b0fd-469b-b59d-7d8c2bfdc3f9)


```python
df.dropna()
```
![image-2](https://github.com/user-attachments/assets/ca1dbebd-798e-4271-9ac9-a4f0efa14f8f)



```python
# TYPE CODE TO FIND MAXIMUM VALUE FROM HEIGHT AND WEIGHT FEATURE
maxHeight = np.max(df["Height"])
maxWeight = np.max(df["Weight"])
print(maxHeight)
print(maxWeight)
```
![image-3](https://github.com/user-attachments/assets/992bbc91-d89f-48ac-84df-b22f1995fbbf)



```python
from sklearn.preprocessing import MinMaxScaler
#Perform minmax scaler
dfmms = df.copy()
scaler = MinMaxScaler()
dfmms[['Height', 'Weight']]=scaler.fit_transform(df[["Height", "Weight"]])
dfmms
```
![image-4](https://github.com/user-attachments/assets/8c45a825-f172-48df-94ab-21e1b2f33929)



```python
from sklearn.preprocessing import StandardScaler
#Perform standard scaler
dfsc=df.copy()
scaler=StandardScaler()
dfsc[["Weight", "Height"]]=scaler.fit_transform(dfsc[["Weight", "Height"]])
dfsc
```
![image-5](https://github.com/user-attachments/assets/70e83d18-156f-4cc6-950e-d7bae9e63b43)



```python
from sklearn.preprocessing import Normalizer
#Perform Normalizer
scaler=Normalizer() 
dfn=df.copy()
dfn[["Weight", "Height"]]=scaler.fit_transform(dfn[["Weight", "Height"]])
dfn
```
![image-6](https://github.com/user-attachments/assets/c3a8d40b-0016-4a1b-a572-2fce34c3fcc9)



```python
from sklearn.preprocessing import MaxAbsScaler
#Perform MaxAbsScaler
mas=MaxAbsScaler() 
dfmas=df.copy() 
dfmas[["Weight", "Height"]]=scaler.fit_transform(dfmas[["Weight", "Height"]])
dfmas
```
![image-7](https://github.com/user-attachments/assets/d74307e1-be12-4de5-b4eb-dbe7665623bc)



```python
from sklearn.preprocessing import RobustScaler
#Perform RobustScaler
rs=RobustScaler() 
dfrs=df.copy() 
dfrs[["Weight", "Height"]]=scaler.fit_transform(dfrs[["Weight", "Height"]])
dfrs
```

![image-8](https://github.com/user-attachments/assets/cbefa4fe-cd87-48e2-a040-ba3e33a0aea5)


```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
```

```python
df=pd.read_csv(r"D:\ds\EXNO-4-DS\titanic_dataset.csv")
df.columns
```
![image-9](https://github.com/user-attachments/assets/b46a4b9b-f653-49ba-8682-d07cd2a75554)


```python
df.shape
```
![image-10](https://github.com/user-attachments/assets/5486f9c4-1483-4b93-81b8-5fb799588dad)



```python
X = df.drop("Survived", axis=1)       # feature matrix
y = df['Survived']
#drop the following columns -"Name", "Sex", "Ticket", "Cabin", "Embarked" and store it in df1
df1=df.drop(["Name", "Sex", "Ticket", "Cabin","Embarked"], axis=1)
```

```python
df1.columns
```
![image-11](https://github.com/user-attachments/assets/47e8c8b9-c3c6-437c-9570-7184cf74ef20)



```python
df1['Age'].isnull().sum()
```
![image-12](https://github.com/user-attachments/assets/66c8379e-c945-45fe-a5a5-53702d9a83ce)



```python
#fill null values of age column using forward fill method
df1["Age"] = df1["Age"].ffill()
df1['Age'].isnull().sum()
```


```python
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
```
![image-14](https://github.com/user-attachments/assets/3f90d3e1-a313-40db-9ce3-1c7a7f7a6879)


```python
#Replace the columns from  ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] to ['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived']
new_cols=['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived']

df1=df1[new_cols]
df1
```
![image-15](https://github.com/user-attachments/assets/93d47956-3903-4dee-8dd7-6069299849c9)



```python
X=df1.iloc[:,0:6]
y=df1.iloc[:,6]
X.columns
```
![image-16](https://github.com/user-attachments/assets/fbca849f-9fd0-4094-9324-ac5f04dee511)




```python
y=y.to_frame()
y.columns
```
![image-17](https://github.com/user-attachments/assets/700ec340-b334-4ccf-9344-c9b9521558fa)



```python
X = df1.drop("Survived", axis=1)       # feature matrix
y = df1['Survived']
```

```python
feature.fit(X,y)
``` 
![image-18](https://github.com/user-attachments/assets/be147b92-ac8e-4512-b0f1-3dff330b85d2)



```python
#perform feature selections techniques
df = df.dropna(subset=['Sex', 'Embarked'])

df['sexle'] = LabelEncoder().fit_transform(df['Sex'])
df['embarkedle'] = LabelEncoder().fit_transform(df['Embarked'])

X_cat = df[['Pclass', 'sexle', 'embarkedle']]
y = df['Survived']

chi_selector = SelectKBest(score_func=chi2, k='all')
chi_selector.fit(X_cat, y)

chi_scores = pd.DataFrame({
    'Feature': X_cat.columns,
    'Chi2 Score': chi_selector.scores_,
    'P-Value': chi_selector.pvalues_
}).sort_values(by='P-Value')

print(chi_scores)
```
![image-19](https://github.com/user-attachments/assets/09bbefa6-320c-4296-8a50-14c1b4500d5b)



# RESULT:
       Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
