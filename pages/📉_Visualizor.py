import Utils
import pandas as pd
from Utils import *


plt.rcdefaults()
plt.style.use('seaborn-darkgrid')

# * creating ususal dictionary for data


age_dict = Utils.age_dict
countries_dict = Utils.countries_dict


# * data processing
age_gender = pd.read_csv('airbnb/age_gender_bkts.csv')
age_values = pd.DataFrame(age_dict.keys(), columns=['age_bucket'])
age_values['values'] = age_dict.values()
age_gender = age_gender.merge(age_values, on='age_bucket', how='inner')
age_gender.sort_values(by='values', ascending=True, inplace=True)


countries = age_gender.country_destination.unique()
gender = age_gender.gender.unique()


# * start of the app
# st.title("Visualizor")

st.write("""
### Visualizor
 - choose a country
 - choose a gender 
 - and see the results
""")

choosen_gender = st.sidebar.selectbox(
    "Select a gender", ['male', 'female', 'both'])
country = st.sidebar.selectbox("Select a country", countries_dict.values())
destination = list(countries_dict.keys())[
    list(countries_dict.values()).index(country)]

#st.write(f"choosen country is : {destination}")
#st.write(f"choosen gender is :{gender}")

# * filtering the data

if choosen_gender == 'both':
    cond2 = age_gender['country_destination'] == destination
    df1 = age_gender[cond2]

else:
    cond1 = age_gender['gender'] == choosen_gender
    cond2 = age_gender['country_destination'] == destination
    df1 = age_gender[cond1 & cond2]


Data, Visuals, BoxPlot = st.tabs(["Data", "Bar plot", "Box Plot"])

Data.subheader("Data")
Data.write(df1)
# * plotting the data

Visuals.subheader("Matplotlib  barplot")
with Visuals.container():
    bar_plot(age_gender, choosen_gender, destination, )

BoxPlot.subheader("Plotly boxplot")
with BoxPlot.container():
    mode = st.selectbox("select a layout", ['vertical', 'horizontal'])
    box_plot(age_gender, choosen_gender, destination, orientation=mode[0])
