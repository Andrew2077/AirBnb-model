from streamlit.logger import get_logger
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib as mpl
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import metrics
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import datetime


from Utils import *


plt.rcdefaults()
plt.style.use('seaborn-darkgrid')
color_list = mpl.rcParams['axes.prop_cycle']
color_list = list(color_list.by_key().values())[0]


LOGGER = get_logger(__name__)

# 🏠 📊 📈 📉 ⚠️🔒 📁  📂 💻 ✈️ 🏁


def run():
    st.set_page_config(
        page_title="Air bnb Data Visualization",
        page_icon="✈️",
        layout="centered",
    )
    # st.write('## this is our Work Documentation ✈️', anchor='#header1')
    # st.header("## Table of Contents")

    



if __name__ == "__main__":
    run()
    st.write("# Airbnb New User Bookings ✈️")


#*  ================================================== Side bar==============================================================
    show_list = st.sidebar.checkbox('show list', False)
    st.markdown("""## $\color{#1f77b4}{\t{AgeGender DataSet}}$
```python
age_gender = pd.read_csv('airbnb/age_gender_bkts.csv') #* reading 1st dataframe
# * dropping year column - not needed
age_gender.drop(['year'], axis=1, inplace=True)
st.age_gender
```
                """)
    
#* =================================================== 1- Age gender DF ===========================================================
    age_gender = pd.read_csv('airbnb/age_gender_bkts.csv')  # * reading 1st dataframe
    # * dropping year column - not needed
    age_gender.drop(['year'], axis=1, inplace=True)
    st.dataframe(age_gender, width=1000)
    st.write('\n')

    st.markdown("""
### After taking a look at the previous sample data,

- The data is clean, no missing values
- year column is redundant
- visualizing the data is going to be mainly done on population_in_thousands

```python
age_gender['population_in_thousands'] #values = 420, unique = 381

```
""")
    st.write('''now let's take a look at the unique values of dataset - apply show list to show output''')
    st.markdown("""
```python
print("Age Bucket : ", age_gender.age_bucket.unique())
print("Countries : ",age_gender.country_destination.unique())
print('Gender :',age_gender.gender.unique())
```

""")
#* =====================================================show list================================================================================================================
    if show_list:
        st.write("Age Bucket : ", list(age_gender.age_bucket.unique()))
        st.write("Countries: ", list(age_gender.country_destination.unique()))
        st.write('Gender :', list(age_gender.gender.unique()))

    st.markdown("""
# Plotting the data -  [$\color{blue}{\t{Visualizer}}$](http://localhost:8501/Visualizor)

- $\color{#d62728}{\t{BarPlot}}$ for population in thousands
  - Filtering by gender
  - populatuin with specfic country - showing the Age bucket wise population
  - populatuin with all countries
  - potential to select specfic age buckets - dropping
- $\color{#d62728}{\t{BoxPlot}}$ for population in thousands
  - females to Country
  - males to Country
  - Both
""")
    
#* ===================================================== bar plot ================================================================================================================
    st.write("## 1 - BarPlots")
    st.markdown("""
let's get our data ready for plotting first
age_dict is a dictionary created in [Utils.py](https://github.com/Andrew2077/AirBnb-model/blob/main/Utils.py)
creating a new df using the dictionary age_dict

```python
age_values = pd.DataFrame(age_dict.keys(), columns=['age_bucket'])
age_values['values'] = age_dict.values()
```
merging the age_gender df with the age_values df giving age_bucket a value from 0 to 19
```python
age_gender = age_gender.merge(age_values, on='age_bucket', how='inner')
age_gender.sort_values(by='values', ascending=True, inplace=True)
```
now let's take a look at the dataframe
```python
age_gender
```
                """)
    age_values = pd.DataFrame(age_dict.keys(), columns=['age_bucket'])
    age_values['values'] = age_dict.values()
    age_gender = age_gender.merge(age_values, on='age_bucket', how='inner')
    age_gender.sort_values(by='values', ascending=True, inplace=True)
    st.dataframe(age_gender)

    st.markdown("""
We have implemented a function that will a barplot for the data on our prefrence
 - for info on the function see Utils.py
 - the plot is done using [matplotlib](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html)

```python
bar_plot(df=age_gender, choosen_gender='male', destination='AU')
```
""")
    bar_plot(df=age_gender, choosen_gender='male', destination='AU')

    st.markdown("""
```python
bar_plot(df=age_gender, choosen_gender='female', destination='AU')
```
""")
    bar_plot(df=age_gender, choosen_gender='female', destination='AU')
    st.markdown("""
```python
bar_plot(df=age_gender, choosen_gender='both', destination='AU')
```
""")
    bar_plot(df=age_gender, choosen_gender='both', destination='AU')
    
#* =================================================== Box plot =============================================================


    st.write("## 2 - BoxPlots")
    st.markdown("""
boxplot is a standart way to how data is distributed but are going to try anther approch first
- df.describe()

```python
age_gender[(age_gender['country_destination'] == 'AU')&
            (age_gender['gender'] =='male')].describe()['population_in_thousands'].round(2)
```
""")
    st.dataframe(age_gender[(age_gender['country_destination'] == 'AU') & (
        age_gender['gender'] == 'male')].describe()['population_in_thousands'])
    st.markdown("""
now let's try a box plot, this time we are using [Plotly](https://plotly.com/python/box-plots/) instead of matplotlib

this is a simple example of box plot function
```python
box_plot_vertical(df=age_gender, choosen_gender='both', destination='AU', orientation='v')
check visualizer for more plots
```
""")
    box_plot(df=age_gender, choosen_gender='both',
                      destination='AU', orientation='v')

    st.markdown("""
## $\color {red} {\t {Insights}}$
- there are Countries prefered than others
- we will be showing why in the next Dataset using Heatmaps 
""")

#* =================================================== 2- Countries DF =============================================================

    st.markdown("""## $\color{#1f77b4}{\t{Countries DataSet}}$
                
#### let's take a look at the data
```python
countries = pd.read_csv('airbnb/countries.csv')
```
                """)
    countries = pd.read_csv('airbnb/countries.csv')
    st.dataframe(countries)
    st.markdown("""
## $\color{blue}{\t{Notes}}$
- ~~lat_destination~~ = $\color{green}{\t{latitude}}$ $\color{red}{\t{redundant}}$ 
- ~~lng_destination~~ = $\color{green}{\t{longitude}}$ $\color{red}{\t{redundant}}$ 
- ~~destination_km2~~ is $\color{red}{\t{redundant}}$ 
- $\color{orange}{\t{levenshtein distance}}$ , In information theory, linguistics, and computer science, the Levenshtein distance is a string metric for measuring the difference between two sequences. Informally, the Levenshtein distance between two words is the minimum number of single-character edits required to change one word into the other.
- **language_levenshtein_distance** =  $\color{lightgreen}{\t{language difficulty}}$ 
- drop $\color{Red}{\t{US}}$ row , as it doesn't have much information to the analysis, and will miss up the calculation of correlation
- giving each country $\color{lightgreen}{\t{Rank by nearst country}}$ of distance from the country to US
- giving each country $\color{lightgreen}{\t{Rank by Travellers}}$ of people traveling to that country
""")

    st.markdown("""
```python
#* let's drop useless columns , gorupby and sort the dataframe
countries = countries.drop(['lat_destination','lng_destination','destination_km2'], axis=1)
population = age_gender.groupby('country_destination')['population_in_thousands'].sum().reset_index()
countries = countries.merge(population, on='country_destination', how='inner')
countries.drop([9], axis=0, inplace=True)

#* renaming the columns
countries.rename(columns={'language_levenshtein_distance': 'language difficulty'}, inplace=True)
countries.rename(columns={'destination_language ': 'destination language'}, inplace=True)
countries.rename(columns={'population_in_thousands': 'Travellers'}, inplace=True)

#* remaping Country names
countries['country_destination'] = countries['country_destination'].map(countries_dict)
countries.reset_index(drop=True, inplace=True)

#* adding new column for ranking 
countries.sort_values(by='Travellers', ascending=False, inplace=True)
countries['Rank by Travellers'] = range(1, len(countries)+1)
countries.sort_values(by='distance_km', ascending=True, inplace=True)
countries['Rank by nearst country'] = range(1, len(countries)+1)

```

now let's take a look at the dataframe

```python
countries
``` 
""")

    # * let's drop useless columns , gorupby and sort the dataframe
    countries = countries.drop(
        ['lat_destination', 'lng_destination', 'destination_km2'], axis=1)
    population = age_gender.groupby('country_destination')[
        'population_in_thousands'].sum().reset_index()
    countries = countries.merge(
        population, on='country_destination', how='inner')
    countries.drop([9], axis=0, inplace=True)

    # * renaming the columns
    countries.rename(
        columns={'language_levenshtein_distance': 'language difficulty'}, inplace=True)
    countries.rename(
        columns={'destination_language ': 'destination language'}, inplace=True)
    countries.rename(
        columns={'population_in_thousands': 'Travellers'}, inplace=True)

    # * remaping Country names
    countries['country_destination'] = countries['country_destination'].map(
        countries_dict)
    countries.reset_index(drop=True, inplace=True)

    # * adding new column for ranking
    countries.sort_values(by='Travellers', ascending=False, inplace=True)
    countries['Rank by Travellers'] = range(1, len(countries)+1)
    countries.sort_values(by='distance_km', ascending=True, inplace=True)
    countries['Rank by nearst country'] = range(1, len(countries)+1)

    st.dataframe(countries)
    
    
#* =================================================== Approch 1 =============================================================

    st.markdown("""
let's try some analysis on the data
## $\color {#2ca02c} {\t {Approch 1 }}$
 - relation between Traverllers and language difficulty
 ```python
Lang_corr = countries.groupby('destination language')[['Travellers', 'language difficulty']].sum(
).sort_values(by='Travellers', ascending=False)
Lang_corr.sort_values(by='language difficulty', ascending=True, inplace=True)
```
let's take a look at our new Language correlation table
```python
Lang_corr
```
    """)
    Lang_corr = countries.groupby('destination language')[['Travellers', 'language difficulty']].sum(
    ).sort_values(by='Travellers', ascending=False)
    Lang_corr.sort_values(by='language difficulty', ascending=True, inplace=True)

    st.dataframe(Lang_corr)
    
    st.markdown("""
let's apply .corr() on dataframe to see the relation betweeen columns
```python
Lang_corr.corr()
```
""")
    st.dataframe(Lang_corr.corr())
    st.markdown("""
now let's plot it on a heatmap
```python
heatmap_plottly(Lang_corr)
```
heatmap_plottly is a function that plots a heatmap of the correlation matrix of the dataframe
check utils.py for more details
""")
    heatmap_plottly(Lang_corr)
    
#* =================================================== Approch 2 =============================================================

    
    st.markdown("""
## $\color {#2ca02c} {\t {Approch 2 }}$
 - relation between Traverllers and Distance from US
 
 ```python
dist_corr = countries[['Travellers','distance_km','country_destination' ]]
dist_corr.set_index('country_destination', inplace=True)
```
let's take a look at our new Distance correlation table
```python
dist_corr
```
 """)
    dist_corr = countries[['Travellers','distance_km','country_destination' ]]
    dist_corr.set_index('country_destination', inplace=True)
    st.dataframe(dist_corr)
    
    st.markdown("""
let's apply .corr() on dataframe to see the relation betweeen columns
```python
dist_corr.corr()
```
""")
    st.dataframe(dist_corr.corr())
    st.markdown("""
now let's plot it on a heatmap
```python
heatmap_plottly(dist_corr)
```
heatmap_plottly is a function that plots a heatmap of the correlation matrix of the dataframe
check utils.py for more details
""")
    heatmap_plottly(dist_corr)
    st.markdown("""
## $\color {red} {\t {Insights}}$
- it's clear that the language difficulty is dominating the analysis
- when the language difficulty increases, the number of travellers decreases 
- distance is a factor in the analysis but it's weight is low compared to language difficulty
""")
    
st.markdown ("""
## $\color {red} {\t {Outlayer- Robusticity }}$
""")