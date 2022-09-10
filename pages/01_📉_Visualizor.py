import Utils
import model_utils
import pandas as pd

from Utils import *
from model_utils import*

from sklearn.ensemble import RandomForestClassifier 


plt.rcdefaults()
plt.style.use('seaborn-darkgrid')

# * creating ususal dictionary for data
st.set_page_config(
    page_title="The Visualizor",
    page_icon="📉",
    layout="wide",
)

age_dict = Utils.age_dict
countries_dict = Utils.countries_dict
country_dict_all = model_utils.country_dict_all


# * data processing
age_gender = pd.read_csv('airbnb/age_gender_bkts.csv')
age_values = pd.DataFrame(age_dict.keys(), columns=['age_bucket'])
age_values['values'] = age_dict.values()
age_gender = age_gender.merge(age_values, on='age_bucket', how='inner')
age_gender.sort_values(by='values', ascending=True, inplace=True)

Train_Data_ori = pd.read_csv('airbnb/train_users_2.csv')
train = Train_Data_ori.copy()


countries = age_gender.country_destination.unique()
gender = age_gender.gender.unique()


# * start of the app
# st.title("Visualizor")

Acc_test = st.sidebar.checkbox('Enable accuracy testing', value=False)
if Acc_test is False :
    choosen_data = st.sidebar.radio("Choose Data", ['Age Gender Data', 'Training Data'])
    if choosen_data == 'Age Gender Data':
        st.write("""
        ### Visualizor
        - choose a country
        - choose a gender 
        - and see the results
        """)
        choosen_gender = st.sidebar.selectbox(
            "Select a gender", ['male', 'female', 'both'])
        country = st.sidebar.selectbox("Select a country", countries_dict.values())
        destination = list(countries_dict.keys())[list(countries_dict.values()).index(country)]

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


    elif choosen_data == 'Training Data':
        st.write("""
        ### Distributions 
        - choose a country
        - AGE for training data age feature
        - FAT for training data first_affiliate_tracked feature
        """)
        # train = Train_Data_ori.copy()
        choosen_feature = st.sidebar.selectbox("Select a feature", ['AGE', 'FAT'])


        country = st.sidebar.selectbox("Select a country", country_dict_all.values())
        destination = list(country_dict_all.keys())[list(country_dict_all.values()).index(country)]

        if choosen_feature == 'AGE':
            distribution_plot_numerical(train, title=choosen_feature, feature2_val = destination )

        elif choosen_feature == 'FAT':
            distribution_plot_categorical(train, title=choosen_feature, feature2_val = destination )

if Acc_test is True :
    train['Month_Reg'] = pd.to_datetime(train.date_account_created).dt.month
    train['Day_Reg'] = pd.to_datetime(train.date_account_created).dt.day
    train['Month_Reg'] = train['Month_Reg'].astype(int)
    train['Day_Reg'] = train['Day_Reg'].astype(int)
    
    train['Month_Booking'] = pd.to_datetime(train.date_first_booking).dt.month
    train['Day_Booking'] = pd.to_datetime(train.date_first_booking).dt.day
    train['Month_Booking'] = train['Month_Booking'].replace(np.NaN, 0).astype(int)
    train['Day_Booking'] = train['Day_Booking'].replace(np.NaN, 0).astype(int)
    
    AGE_manipulate = st.sidebar.radio("Age Imputation method", AGE_method.keys(), index=0)
    FAT_manipulate = st.sidebar.radio("FAT Imputation method", FAT_method.keys(), index=2)
    show_df = st.sidebar.checkbox('show Final Dataframe', value=False)
    Model_selection = st.sidebar.radio("Model Selection", ['Decision Tree Classifier', 'Random Forest', 'XGBoost'])
    # train = pd.read_csv('train_users_2.csv')


    
    train_modified_age = fill_missing_numerical(df = train['age'], method = AGE_method[AGE_manipulate])
    indices = train_modified_age.index
    train = train.iloc[indices]
    train['age'] = train_modified_age
    train['first_affiliate_tracked'] = fill_missing_categorical(df = train['first_affiliate_tracked'], method = AGE_method[FAT_manipulate])

    train = discrete_categories(train, cols)
    train.drop(['id', 'date_account_created','timestamp_first_active', 'date_first_booking'], axis=1, inplace=True)
    testing_part = unbaised_sample(train)
    training_part = train.iloc[train.index.delete(testing_part.index)]

    ytrain = training_part['country_destination']
    xtrain = training_part.drop(['country_destination'], axis=1)
    # xtrain = discrete_categories(xtrain, cols)

    ytest = testing_part['country_destination']
    xtest = testing_part.drop(['country_destination'], axis=1)
    
    if show_df is True:
        st.dataframe(train)
    
    if Model_selection == 'Decision Tree Classifier':
        score = str(round(print_score(DecisionTreeClassifier(), xtrain, ytrain, xtest, ytest), 4)*100)
    elif Model_selection == 'Random Forest':
        score = str(round(print_score(RandomForestClassifier(criterion = 'entropy'), xtrain, ytrain, xtest, ytest), 4)*100)
    original_title = f'<p style="color:#1f77b4; font-size: 25px;">Accuracy Score is {score[0:5]}%</p>'
    st.markdown(original_title, unsafe_allow_html=True)
  
    #st.write('score: ', print_score(DecisionTreeClassifier(), x_all, y_all))