from Utils import*
from model_utils import*
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
     page_title="AirBnb prediction model",
     page_icon="💻",
     layout="wide",
     initial_sidebar_state="expanded",
)
#* ##################################################### reading data #######################################################
st.markdown("""## $\color{#1f77b4}{\t{Training DataSet}}$ """)
st.markdown("""
Let's first take a look at our training data 
```python
train = pd.read_csv('airbnb/train_users_2.csv')
```
let's take a look at it's shape
```python
train.shape
```
""")
train = pd.read_csv('airbnb/train_users_2.csv')
st.write(train.shape)

st.markdown("""showing training df head
```python
train.head(10)
            """)
st.dataframe(train.head(10))
info = train.isna().sum()
st.markdown("""
now let's take a look at null values
```python
train.isna().sum()
```
""")
st.write(info)

#* ####################################################### data preprocessing - age #######################################################
st.markdown("""
## $\color {#2ca02c} {\t {Age - numerical }}$
```python
train['age'].dtype
```
""")
st.write(train['age'].dtype)
st.markdown("""
```python
train['first_affiliate_tracked'].isna().value_counts()
```          
""")
st.write(train['age'].isna().value_counts())


st.markdown("""
#### **there are many ways to handle numerical missing values**
#### let's try few of them that will work on numeric values
Using fill_missing_numerical() function to fill missing numerical values
with the ability to adjust the IQR ratio and the number of outlayers to be removed
```python
fill_missing_numerical(df, methode, IQR_ratio=5, show_outlayers_num=False, show_dist_plot=False, show_values_range=False)
``` 
check fill_missing_numerical() in [model_utils.py](https://github.com/Andrew2077/AirBnb-model/blob/main/model_utils.py) for more details

there are 4 imputation methods that are are available in the implementation 

- imputing with the mean
```python
df = df.replace(np.NaN, round(df.mean(), 0))
```
- imputing with the median
```python
df = df.replace(np.NaN, df.median())
```
- imputing with backward filling
```python
df = df.fillna(method='bfill')
```
- imputing with forward filling
```python
df = df.fillna(method='ffill')
```
check [models_utils.py](https://github.com/Andrew2077/AirBnb-model/blob/main/model_utils.py) for more details on how it was implemented
and let's see how will this effect the distribution of the data

```python
distribution_plot_numerical(train, title='Age', feature2_val='all', bins=120, feature1='age', feature2='country_destination')
```
""")

distribution_plot_numerical(train, title='Age', feature2_val='all', bins=120, feature1='age', feature2='country_destination')
#* ####################################################### Insights 1  #######################################################
st.markdown("""
## $\color {red} {\t {Insights}}$
- althought it's clear that Backward filling and Forward filling are the best methods to impute missing values as they do not affect the distribution of the data
- but that doesn't mean that it will give the highest accuracy on the model of predicting the data
- it will be more clear with Decision Tree Classifier model
### $\color {red} {\t {Notes}}$
- there are more mothods to handle missing values 
     - imputing with the random values in the range of the data
     - dropping rows with missing values
     - creating a model to predict missing values
---
---
""")


#* ####################################################### data preprocessing - FAT #######################################################

st.markdown("""
## $\color {#2ca02c} {\t {FAT - categorical}}$ 
FAT is the short for **"first_affiliate_tracked"** which is the 2nd feature that has missing values
since we already know that that Age data were numeric, let's see what is the type of FAT data
```python
train['first_affiliate_tracked'].value_counts()
train['first_affiliate_tracked'].dtype
""")

st.write(train['first_affiliate_tracked'].value_counts())
st.write(train['first_affiliate_tracked'].dtype)


st.markdown("""
```python
train['first_affiliate_tracked'].isna().value_counts()
""")
st.write(train['first_affiliate_tracked'].isna().value_counts())

st.markdown("""   
### **there are many ways to handle categorical missing values**
### let's try few of them that will work on categorical values
Using fill_missing_numerical() function to fill missing categorical values

```python
fill_missing_categorical(df, method)
``` 
check fill_missing_categorical() in [model_utils.py](https://github.com/Andrew2077/AirBnb-model/blob/main/model_utils.py) for more details

there are 4 imputation methods that are available in the implementation 
- imputing with the Mode
```python
df = df.replace(np.NaN, round(df.mean(), 0))
```
- imputing with Random Value
```python
import random
vals = list(df.dropna().unique())
df = df.replace(np.NaN, random.choice(vals))
```
- imputing with backward filling
```python
df = df.fillna(method='bfill')
```
- imputing with forward filling
```python
df = df.fillna(method='ffill')
```
check [models_utils.py](https://github.com/Andrew2077/AirBnb-model/blob/main/model_utils.py) for more details on how it was implemented
and let's see how will this effect the distribution of the data

```python
distribution_plot_cate(df, title='FAT', country='all', bins=12, feature1 = 'first_affiliate_tracked', feature2 ='country_destination')
```
""")
distribution_plot_categorical(train, title='FAT', country='all', bins=12, feature1 = 'first_affiliate_tracked', feature2 ='country_destination')


#* ####################################################### Insights 2  #######################################################
st.markdown("""
## $\color {red} {\t {Insights}}$
- it's hard to tell if filling the missing values messed up the distribution of the data
- even if it did, it's dominated by one class
### $\color {red} {\t {Notes}}$
- Same as numerical data, there are few more methods to handle categorical missing values
     - dropping rows with missing values
     - creating a model to predict missing values
---
---
""")

st.markdown("""
## $\color {#2ca02c} {\t {DFB - TimeSeries }}$
DFB is the short for **"date_first_booking"** which is the 3nd feature and has missing values
times seres data type 
### $\color {red} {\t {Stay Tuned }}$

---
---
""")

st.markdown("""
# $\color {blue} {\t {Predicive-Model }}$
---
let's choose our features that we want to use for our model
```python
train.head(3)
```""")
st.write(train.head(3))
st.markdown("""
before creating the model, let's save our data intries to check how many we lost in the process
```python
shape_before = train.shape[0]
```
""")
shape_before = train.shape[0]
st.markdown("""
let's drop unwanted features/columns
```python
train = train.drop(['id', 'date_account_created',
     'timestamp_first_active', 'date_first_booking'], axis=1, )
```
$\color {red} {\t {note}}$ : the drop features step is done inside a function called discrete_categories(): in model_utils.py
## $\color {#2ca02c} {\t {Decision-Tree-Classifier }}$
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
```
to turn off unnecessary warnings
```python
import warnings
warnings.filterwarnings('ignore')
```

now let's create our model
althought it's not the best thing to do, but always make sure you're working on the Orginial data
```python
train = pd.read_csv('airbnb/train_users_2.csv')
```
## workflow
1. handle numerical missing values
2. handle categorical missing values
3. discretize categorical features
4. create model
     
#### 1. handling numerical missing values
```python
train_modified_age = fill_missing_numerical(df = train['age'], methode = 'mean')
```
it's built to handle also the outliers using IQR method, set IQR_ratio to adjust outliers tolerence
so some data will be dropped ~outliers~
to update the dataframe with the new values, some indcies are needed
```python
indices = train_modified_age.index
train = train.iloc[indices]
train['age'] = train_modified_age

### #2. handling categorical missing values
this step is easy cause categorical doesn't have outliers
we directly update categorical dataframe with the new values
```python
train['first_affiliate_tracked'] = fill_missing_categorical(df = train['first_affiliate_tracked'], method = 'ffill')

### 3. discretize categorical features
using the function discrete_categories() in model_utils.py to discretize categorical features
```python
modified_df = discrete_categories(train, cols)
```
### 4. create model
spliting the data into x and y
```python
x_all = modified_df[modified_df.columns[:-1]]
y_all = modified_df[modified_df.columns[-1]]
```
then call the function print_score() in model_utils.py to print the model score
```python
print_score(DecisionTreeClassifier(), x_all, y_all)
```

""")
# train = pd.read_csv('train_users_2.csv')
train_modified_age = fill_missing_numerical(df = train['age'], method = 'mean')
indices = train_modified_age.index
train = train.iloc[indices]
train['age'] = train_modified_age
train['first_affiliate_tracked'] = fill_missing_categorical(df = train['first_affiliate_tracked'], method = 'ffill')
        
modified_df = discrete_categories(train, cols)

x_all = modified_df[modified_df.columns[:-1]]
y_all = modified_df[modified_df.columns[-1]]

st.write('score: ', print_score(DecisionTreeClassifier(), x_all, y_all))

