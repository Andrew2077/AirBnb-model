import Utils
from Utils import *


plt.rcdefaults()
plt.style.use('seaborn-darkgrid')

# * creating ususal dictionary for data


age_dict = Utils.age_dict
countries_dict = Utils.countries_dict


# * data processing
age_gender = pd.read_csv('airbnb/age_gender_bkts.csv')
age_gender.drop(['year'], axis=1, inplace=True)

countries = age_gender.country_destination.unique()
gender = age_gender.gender.unique()

age_values = pd.DataFrame(age_dict.keys(), columns=['age_bucket'])
age_values['values'] = age_dict.values()
age_gender = age_gender.merge(age_values, on='age_bucket', how='inner')


# * start of the app
# st.title("Visualizor")

st.write("""
### Visualizor
 - choose a country
 - choose a gender 
 - and see the results
""")

gender = st.sidebar.selectbox("Select a gender", gender)
country = st.sidebar.selectbox("Select a country", countries_dict.values())
destination = list(countries_dict.keys())[
    list(countries_dict.values()).index(country)]

#st.write(f"choosen country is : {destination}")
#st.write(f"choosen gender is :{gender}")

# * filtering the data
cond1 = age_gender['gender'] == gender
cond2 = age_gender['country_destination'] == destination
df1 = age_gender[cond1 & cond2]
df1.sort_values(by='values', ascending=True, inplace=True)

Visuals, Data , BoxPlot= st.tabs(["Chart", "Data","Box Plot"])

Data.subheader("Data")
Data.write(df1)
# * plotting the data

Visuals.subheader("Visualization")
fig, ax = plt.subplots(figsize=(8, 5))

bars = plt.bar(
    x=df1['values']*2,
    height=df1['population_in_thousands'],
    width=1.3,
)

plt.xticks(df1['values']*2, df1['age_bucket'])

x = plt.gca().xaxis
for item in x.get_ticklabels():
    item.set_fontsize(10)
    item.set_rotation(45)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)

bar_color = bars[0].get_facecolor()
height = (bars[0].get_height()/100)*2

for bar in bars:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + height,
        int(round(bar.get_height(), 0)),
        horizontalalignment='center',
        color=bar_color,
        fontsize=9,
        weight='bold'
    )

ax.set(facecolor='lightgray')
fig.set(facecolor='lightgray')

plt.grid(axis='x', visible=False)
ax.get_yaxis().set_ticks([])
fig.tight_layout()
ax.margins(x=0.01)
ax.margins(y=0.005)

#plt.legend(['Population'], loc='best', fontsize=10)

plt.title("Flights of {}s to {}".format(gender, country), fontsize=23, pad=35)
plt.xlabel('Age Bucket', fontsize=17,)
plt.ylabel('Population in Thousands', fontsize=17,)
# plt.show()

Visuals.pyplot(fig)

st.set_option('deprecation.showPyplotGlobalUse', False)
