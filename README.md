## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1: Read the given Data.

STEP 2: Clean the Data Set using Data Cleaning Process.

STEP 3: Apply Feature Encoding for the feature in the data set.

STEP 4: Apply Feature Transformation for the feature in the data set.

STEP 5: Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
```
```
df=pd.read_csv('/content/Encoding Data (2).csv')
```
```
df
```
![image](https://github.com/user-attachments/assets/70213551-d09e-4c77-9ee5-fd0f7ca7b87e)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
```

```
pm=['Hot','Warm','Cold']
```
```
e1=OrdinalEncoder(categories=[pm])
```
```
e1.fit_transform(df[['ord_2']])
```
![image](https://github.com/user-attachments/assets/de84f215-719f-4b4e-afdb-c6d8d01fd78d)

```
df['bo2']=e1.fit_transform(df[['ord_2']])
df
```
![image](https://github.com/user-attachments/assets/dbd4cbd9-df04-4ebd-87a3-ae708b9c0cea)

```
le=LabelEncoder()
```
```
dfc=df.copy()
```
```
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
```
```
dfc
```

![image](https://github.com/user-attachments/assets/ceff7393-c6b1-4f83-8f75-4a45918d7688)

```
from sklearn.preprocessing import OneHotEncoder
```
```
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
```
```
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
```
df2=pd.concat([df2,enc],axis=1)
```
```
df2
```
![image](https://github.com/user-attachments/assets/bd278ddc-351b-4c62-8fdb-c60e702ded8c)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/user-attachments/assets/11d1e4b4-4205-48aa-8ed6-c2e368fd30e5)

```
from category_encoders import BinaryEncoder
```

```
df=pd.read_csv('/content/data (2).csv')
```
```
df
```

![image](https://github.com/user-attachments/assets/ccbc8288-3ffc-42e7-a266-be79e83c26ce)

```
be=BinaryEncoder()
```
```
nd=be.fit_transform(df['Ord_2'])
```
```
dfb=pd.concat([df,nd],axis=1)
```
```
dfb1=df.copy()
```
```
dfb
```

![image](https://github.com/user-attachments/assets/5adfbe83-e18c-449c-91ed-0f94e381310b)

```
from category_encoders import TargetEncoder
```
```
te=TargetEncoder()
```
```
cc=df.copy()
```
```
new=te.fit_transform(X=cc['City'],y=cc['Target'])
```
```
cc=pd.concat([cc,new],axis=1)
```
```
cc
```

![image](https://github.com/user-attachments/assets/752a5e39-93f2-46b8-93f5-0d39f8628d6b)

```
import pandas as pd
from scipy import stats
import numpy as np

```
```
df=pd.read_csv('/content/Data_to_Transform (1).csv')
```
```
df
```

![image](https://github.com/user-attachments/assets/53894387-9bff-43fa-9866-5a5a18b36659)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/d5977e3c-aafd-48ab-8513-57f09b3e0780)

```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/026f555b-1cbd-418f-865f-b103b9393354)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/8a4239d3-c217-4ad4-8b95-705765ef1ff8)

```
np.sqrt(df["Highly Negative Skew"])
```
![image](https://github.com/user-attachments/assets/ab521821-a19f-4087-a3ed-3fc0816e7af5)

```
np.square(df["Highly Negative Skew"])
```
![image](https://github.com/user-attachments/assets/8dd6165f-4a6f-4060-9956-c3304e180df0)

```
df["Highly Positive Skew"],parameterss=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/d97e8e94-9de9-40dd-8141-d0687f1e6ef4)

```
df["Moderate Negative Skew_yeojohnson"],parameterss=stats.yeojohnson(df["Moderate Negative Skew"])
```

```
df.skew()
```
![image](https://github.com/user-attachments/assets/1533122c-9454-46b8-bb5b-31b7e1a38522)

```
from sklearn.preprocessing import QuantileTransformer
```
```
qt=QuantileTransformer(output_distribution='normal')
```
```
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```
```
df
```
![image](https://github.com/user-attachments/assets/e3b6e24b-347c-47ee-8d9c-83a84cd8ffa1)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/d9e3d1a9-09ac-41dc-a3db-f4560b6a8834)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/35240e32-7040-4fa8-ab6f-d1862bcf5b6c)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```

```
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```
```
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/f3a55f37-70af-46e5-8e1b-e840273d89b2)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/1bebc202-2ba1-439e-8cbb-a9c4a0eaea7e)

```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/5b89f39a-4422-49ec-857b-f725b5444c5b)

```
dt=pd.read_csv('/content/titanic_dataset (2).csv')
```
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
```
```
sm.qqplot(dt["Age"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/7d85f449-371b-477f-ad4c-eb22ed7f5a3a)

```
sm.qqplot(dt["Age_1"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/337a086e-6dbe-4437-ba87-d10a41cef303)
# RESULT:
The data was successfully read, feature encoding and transformation were performed, and the processed data was saved to a file named DS Ex No 3.

       
