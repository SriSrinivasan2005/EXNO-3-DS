## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

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
ds=pd.read_csv("Encoding Data.csv")
ds
```

<img width="539" height="542" alt="Screenshot 2025-09-30 104954" src="https://github.com/user-attachments/assets/4d0edca1-3014-42c4-a714-e581fd45dec8" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
a=['Hot','Warm','Cold']
a1=OrdinalEncoder(categories=[a])
a1.fit_transform(ds[["ord_2"]])
```
<img width="681" height="355" alt="Screenshot 2025-09-30 105005" src="https://github.com/user-attachments/assets/112429c0-0dcb-498d-8fa4-4cef32ed6693" />

```
ds['bo2']=a1.fit_transform(ds[["ord_2"]])
ds
```
<img width="605" height="525" alt="Screenshot 2025-09-30 105020" src="https://github.com/user-attachments/assets/fccba83d-f760-417d-94d3-7c3b15a90f47" />


```
l=LabelEncoder()
dfc=ds.copy()
dfc['ord_2']=l.fit_transform(dfc['ord_2'])
dfc
```
<img width="599" height="579" alt="Screenshot 2025-09-30 105030" src="https://github.com/user-attachments/assets/54493127-a311-4e6f-9419-4fb33cbd4c6f" />

```
from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder(sparse_output=False)
ds2=ds.copy()
enc=pd.DataFrame(oh.fit_transform(ds2[["nom_0"]]))
ds2=pd.concat([ds2,enc],axis=1)
ds2
```
<img width="650" height="616" alt="Screenshot 2025-09-30 105043" src="https://github.com/user-attachments/assets/3b757cd2-b55f-44ab-b9ba-135f0fefca54" />


```
pd.get_dummies(ds2,columns=["nom_0"])
```
<img width="979" height="493" alt="Screenshot 2025-09-30 105052" src="https://github.com/user-attachments/assets/8f212271-d7e4-4012-a045-ab702917c00d" />

```
pip install --upgrade category_encoders
```
<img width="1574" height="474" alt="Screenshot 2025-09-30 105114" src="https://github.com/user-attachments/assets/b9112d86-466f-4bb8-99dd-a78844a07a2f" />


```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="768" height="557" alt="Screenshot 2025-09-30 105124" src="https://github.com/user-attachments/assets/2bba3b08-efb7-4768-b47a-3992720644d8" />

```
b=BinaryEncoder()
nd=b.fit_transform(df['Ord_2'])
df
```
<img width="761" height="547" alt="Screenshot 2025-09-30 105139" src="https://github.com/user-attachments/assets/b2bd24d0-367c-4ce3-8786-1f14db8429bc" />

```
db=pd.concat([df,nd],axis=1)
db
```
<img width="977" height="517" alt="Screenshot 2025-09-30 105147" src="https://github.com/user-attachments/assets/f04e4994-458b-4f20-80b9-7e209b3e2555" />

```
from category_encoders import TargetEncoder
t=TargetEncoder()
CC=df.copy()
new=t.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="835" height="622" alt="Screenshot 2025-09-30 105200" src="https://github.com/user-attachments/assets/a7f11c53-9487-41fa-9e28-c81be53af72e" />

```
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="1055" height="648" alt="Screenshot 2025-09-30 105210" src="https://github.com/user-attachments/assets/e9a3572e-d070-4f1d-ba72-d1de997bce77" />

```
df.skew()
```
<img width="506" height="313" alt="Screenshot 2025-09-30 105219" src="https://github.com/user-attachments/assets/fb8b9b98-643a-4bf8-aaa5-db62297989de" />

```
np.log(df["Highly Positive Skew"])
```
<img width="601" height="619" alt="Screenshot 2025-09-30 105227" src="https://github.com/user-attachments/assets/83d03f87-b23c-4201-bae1-0b4c6c748194" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="717" height="623" alt="Screenshot 2025-09-30 105236" src="https://github.com/user-attachments/assets/b5bc8808-68dc-411b-bf4c-0b5ae40ffd3f" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="628" height="614" alt="Screenshot 2025-09-30 105250" src="https://github.com/user-attachments/assets/423ea32b-5316-42aa-8c0a-852413cbbb14" />

```
np.square(df["Highly Positive Skew"])
```
<img width="593" height="624" alt="Screenshot 2025-09-30 105301" src="https://github.com/user-attachments/assets/c981c6b5-a621-45f6-8e84-3f24899812a6" />


```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1318" height="610" alt="Screenshot 2025-09-30 105315" src="https://github.com/user-attachments/assets/ebaadc39-6855-4c01-9e99-113f2da0337c" />

```
df.skew()
```
<img width="540" height="346" alt="Screenshot 2025-09-30 105322" src="https://github.com/user-attachments/assets/88ecbba6-465e-4566-9090-1af67214c54b" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="1064" height="409" alt="Screenshot 2025-09-30 105329" src="https://github.com/user-attachments/assets/eae1d457-2db2-4a8b-a6be-c74ab9703cf2" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1729" height="661" alt="Screenshot 2025-09-30 105348" src="https://github.com/user-attachments/assets/c69cfc76-53b0-4e5c-bb59-f249e7aaa11d" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="915" height="706" alt="Screenshot 2025-09-30 105356" src="https://github.com/user-attachments/assets/1bebb66c-2b20-472b-b29c-63ba1f5706e1" />

```
sm.qqplo![Uploading Screenshot 2025-09-30 105356.png…]()
t(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="886" height="640" alt="Screenshot 2025-09-30 105402" src="https://github.com/user-attachments/assets/6fefbe1b-21ba-41b1-bf2f-347a352bdf54" />

```
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="856" height="683" alt="Screenshot 2025-09-30 105410" src="https://github.com/user-attachments/assets/7e42364e-35b1-4470-944a-f8f61ac1b1d1" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="904" height="660" alt="Screenshot 2025-09-30 105417" src="https://github.com/user-attachments/assets/a9aa0b74-16e1-489e-ae34-05936e443fd6" />

```
df1=pd.read_csv("titanic_dataset.csv")
df1
```
<img width="1556" height="590" alt="Screenshot 2025-09-30 105431" src="https://github.com/user-attachments/assets/9ffcf08e-6d9a-4f9d-913c-219d6d25821b" />

```
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df1["Age_1"]=qt.fit_transform(df1[["Age"]])
sm.qqplot(df1['Age'],line='45') 
plt.show()
```
<img width="919" height="680" alt="Screenshot 2025-09-30 105438" src="https://github.com/user-attachments/assets/b7c3e176-5a10-4a17-97f9-c52385598dd4" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="874" height="642" alt="Screenshot 2025-09-30 105446" src="https://github.com/user-attachments/assets/5b347ce8-ac28-47ad-95c6-d08d0ba7c9f5" />


## RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
