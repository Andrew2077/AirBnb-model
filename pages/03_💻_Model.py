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
#* ##################################################### Main plan #######################################################
st.markdown("""## $\color{#69109c}{\t{Training DataSet}}$ 
            
- ##### $\color{#439c10}{\t{1- Features Adjustments}}$ 
- ##### $\color{#439c10}{\t{2- Handling Missing Values}}$ 
- ##### $\color{#439c10}{\t{3- Model Training}}$

""")

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

#* COlumns info 
st.markdown("""
### Columns to be dropped
- id 
```python
train.drop(['id'], axis=1, inplace=True)) # will be dropped later on
```   
### Columns to be adjusted
- all the other categroical cols 
- Time series cols 

### Columns that have missing values
- first_affiliate_tracked 
- age 
            """)



info = train.isna().sum()
st.markdown("""
now let's take a look at null values
```python
train.isna().sum()
```
""")
st.write(info)

#* ##################################################### 1- Features Adjustments #######################################################
#* 1- signup flow
st.markdown("""
### $\color{#ac0020}{\t{1- Features Adjustment}}$ 
#### $\color{#00ac8c}{\t{SignFlow}}$  the page a user came to signup up from
```python
train.signup_flow.value_counts()
```
""")
st.write(train.signup_flow.value_counts())

st.markdown("""
it seems that the home page 0 is the most common page, replace low frequency pages with 0
```python
# replace low frequency pages with 0
singupfollow_replaced = list(train.signup_flow.value_counts()[train.signup_flow.value_counts() < 400].index)  
# the most common page
singupfollow_replacing_value = train.signup_flow.value_counts().index[0] 
# replacing
train['signup_flow'].replace(singupfollow_replaced, singupfollow_replacing_value ,inplace=True) 
```
            """)
singupfollow_replaced = list(train.signup_flow.value_counts()[train.signup_flow.value_counts() < 400].index)
singupfollow_replacing_value = train.signup_flow.value_counts().index[0]
train['signup_flow'].replace(singupfollow_replaced, singupfollow_replacing_value ,inplace=True)
st.write(train.signup_flow.value_counts())

#* language
st.markdown("""
#### $\color{#00ac8c}{\t{Language}}$, the language of the user
```python
train.language.value_counts()
```
""")
st.write(train.language.value_counts())

st.markdown("""
replace low frequency languages with `other`
```python
# replace low frequency pages with 0
language_replaced = list(train.language.value_counts()[train.language.value_counts() < 100].index)
# the most common page
language_replacing_value = 'other'
# replacing
train['language'].replace(language_replaced, language_replacing_value ,inplace=True)
```
""")
language_replaced = list(train.language.value_counts()[train.language.value_counts() < 100].index)
language_replacing_value = 'other'
train['language'].replace(language_replaced, language_replacing_value ,inplace=True)
st.write(train.language.value_counts())

#* affiliate_provider
st.markdown("""
#### $\color{#00ac8c}{\t{affiliate - provider}}$, is a person or health care facility paid by your health care plan to provide service to you.
```python
train.affiliate_provider.value_counts()
```
""")
st.write(train.affiliate_provider.value_counts())

st.markdown("""
replace low frequency affiliate providers with `other`
```python
# replace low frequency pages with 0
affiliate_provider_replaced = list(train.affiliate_provider.value_counts()[train.affiliate_provider.value_counts() < 300].index)
# the most common page
affiliate_provider_replacing_value = 'other'
# replacing
train['affiliate_provider'].replace(affiliate_provider_replaced, affiliate_provider_replacing_value ,inplace=True)
```
""")

affiliate_provider_replaced = list(train.affiliate_provider.value_counts()[train.affiliate_provider.value_counts() < 300].index)
affiliate_provider_replacing_value = 'other'
train['affiliate_provider'].replace(affiliate_provider_replaced, affiliate_provider_replacing_value ,inplace=True)

st.write(train.affiliate_provider.value_counts())

#* gender
st.markdown("""
#### $\color{#00ac8c}{\t{Gender}}$
```python
train.gender.value_counts()
```
""")
st.write(train.gender.value_counts())
st.markdown("""
replace `-unknown-` with `prefere not telling`
```python
train['gender'].replace('-unknown-', 'prefere not telling' ,inplace=True)
```
""")
train['gender'].replace('-unknown-', 'prefere_not_telling' ,inplace=True)



#* ####################################################### data preprocessing - age #######################################################
st.markdown("""
### $\color{#ac0020}{\t{2- Handling Missing Values}}$ 
#### $\color {#00ac8c} {\t {Age - numerical }}$
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
#### $\color {#00ac8c} {\t {FAT - categorical}}$ 
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
df = df.replace(np.NaN, df.mode()[0])
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
distribution_plot_categorical(train, title='FAT', feature2_val='all', bins=12, feature1 = 'first_affiliate_tracked', feature2 ='country_destination')


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
### $\color {#00ac8c} {\t {DFB - TimeSeries }}$
#### date_account_created
```python
train.date_account_created.dtypes
```

""")
st.write(train.date_account_created.dtypes)

st.markdown("""
```python
# train['Year'] = pd.to_datetime(train.date_account_created).dt.year
train['Month'] = pd.to_datetime(train.date_account_created).dt.month
train['Day'] = pd.to_datetime(train.date_account_created).dt.day

train['Month_Reg'] = train['Month_Reg'].astype(int)
train['Day_Reg'] = train['Day_Reg'].astype(int)
```
after that we can drop the date_account_created column
""")

# train['Year'] = pd.to_datetime(train.date_account_created).dt.year
train['Month_Reg'] = pd.to_datetime(train.date_account_created).dt.month
train['Day_Reg'] = pd.to_datetime(train.date_account_created).dt.day
train['Month_Reg'] = train['Month_Reg'].astype(int)
train['Day_Reg'] = train['Day_Reg'].astype(int)

st.markdown("""
#### date_first_booking
we don't have to fill the missing values of this column, as nan can be represented as 0
```python 
train.date_first_booking.dtype
```
""")
st.write(train.date_first_booking.dtype)

st.markdown("""
```python
train['Month_Booking'] = pd.to_datetime(train.date_first_booking).dt.month
train['Day_Booking'] = pd.to_datetime(train.date_first_booking).dt.day
``` 
now let's replace the missing values with 0

```python
train['Month_Booking'] = train['Month_Booking'].replace(np.NaN, 0).astype(int)
train['Day_Booking'] = train['Day_Booking'].replace(np.NaN, 0).astype(int)
```
after that we can drop the date_first_booking column

#### timestamp_first_active

can be used as feature but i've removed it from the model
cause the user may have visited the website by mistake back then

---
---
""")
train['Month_Booking'] = pd.to_datetime(train.date_first_booking).dt.month
train['Day_Booking'] = pd.to_datetime(train.date_first_booking).dt.day
train['Month_Booking'] = train['Month_Booking'].replace(np.NaN, 0).astype(int)
train['Day_Booking'] = train['Day_Booking'].replace(np.NaN, 0).astype(int)

show_list = st.sidebar.checkbox('Show Columns', False)

if show_list:
     st.write(list(train.columns))

st.markdown("""
### $\color{#ac0020}{\t{3- Model Training}}$ 
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
#### $\color {#2ca02c} {\t {Decision-Tree-Classifier }}$
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
### 4. split data into train and test
```python
# drop unwanted features
train.drop(['id', 'date_account_created','timestamp_first_active', 'date_first_booking'], axis=1, inplace=True)

# creating a test sample with 10% while keeping the same distribution
testing_part = unbaised_sample(train)
training_part = train.iloc[train.index.delete(testing_part.index)]

# finializing the data
ytrain = training_part['country_destination']
xtrain = training_part.drop(['country_destination'], axis=1)

ytest = testing_part['country_destination']
xtest = testing_part.drop(['country_destination'], axis=1)
```
### 5. create model
call the function print_score() in model_utils.py to print the model score
```python
print_score(DecisionTreeClassifier(), xtrain, ytrain, xtest, ytest)
```

""")
# train = pd.read_csv('train_users_2.csv')

train_modified_age = fill_missing_numerical(df = train['age'], method = 'mean')
indices = train_modified_age.index
train = train.iloc[indices]
train['age'] = train_modified_age
train['first_affiliate_tracked'] = fill_missing_categorical(df = train['first_affiliate_tracked'], method = 'bfill')



train = discrete_categories(train, cols)
train.drop(['id', 'date_account_created','timestamp_first_active', 'date_first_booking'], axis=1, inplace=True)
testing_part = unbaised_sample(train)
training_part = train.iloc[train.index.delete(testing_part.index)]

ytrain = training_part['country_destination']
xtrain = training_part.drop(['country_destination'], axis=1)
# xtrain = discrete_categories(xtrain, cols)

ytest = testing_part['country_destination']
xtest = testing_part.drop(['country_destination'], axis=1)
# xtest = discrete_categories(xtest, cols)


## retrieve data to predict 

# st.write('score: ', print_score(DecisionTreeClassifier(), xtrain, ytrain, xtest, ytest))
score = str(round(print_score(DecisionTreeClassifier(), xtrain, ytrain, xtest, ytest), 4)*100)
original_title = f'<p style="color:#1f77b4; font-size: 25px;">Accuracy Score is {score[0:5]}%</p>'
st.markdown(original_title, unsafe_allow_html=True)

test_data = pd.read_csv('airbnb/test_users.csv')
test_data['Month_Reg'] = pd.to_datetime(test_data.date_account_created).dt.month
test_data['Day_Reg'] = pd.to_datetime(test_data.date_account_created).dt.day
test_data['Month_Reg'] = test_data['Month_Reg'].astype(int)
test_data['Day_Reg'] = test_data['Day_Reg'].astype(int)

test_data['Month_Booking'] = pd.to_datetime(test_data.date_first_booking).dt.month
test_data['Day_Booking'] = pd.to_datetime(test_data.date_first_booking).dt.day
test_data['Month_Booking'] = test_data['Month_Booking'].replace(np.NaN, 0).astype(int)
test_data['Day_Booking'] = test_data['Day_Booking'].replace(np.NaN, 0).astype(int)
    


test_modified_age = fill_missing_numerical(df = test_data['age'], method = 'mean')
indices = test_modified_age.index
test_data = test_data.iloc[indices]
test_data['age'] = test_modified_age
test_data['first_affiliate_tracked'] = fill_missing_categorical(df = test_data['first_affiliate_tracked'], method = 'bfill')

test_data = discrete_categories(test_data, cols)
test_data.drop(['id', 'date_account_created','timestamp_first_active', 'date_first_booking'], axis=1, inplace=True)



st.markdown(""" let's see how the model works on the test data
```python
train = pd.read_csv('airbnb/train_users_2.csv')
train = train.iloc[testing_part.index]
the score is 
```
""")

# st.write('score: ', print_score(DecisionTreeClassifier(), xtrain, ytrain, xtest, ytest))
test_score = str(round(print_score(DecisionTreeClassifier(), xtrain, ytrain, xtest, ytest), 4)*100)
original_title = f'<p style="color:#1f77b4; font-size: 25px;">Accuracy Score is {test_score[0:5]}%</p>'
st.markdown(original_title, unsafe_allow_html=True)

st.markdown("""
## $\color {red} {\t {Insights}}$
- imputing age with Mean and FAT with backward filling, has performed the best so far
- you can check scores on testing tap 

### $\color {red} {\t {Notes}}$
- why Decision Tree classifier ?
     - it's a good model to use for classification problems
     - we have 10 features to work with, so it's a faster model than other models
     - there are other models that can predict much better than Decision Tree classifier, but will need lot of work
---
---
""")

