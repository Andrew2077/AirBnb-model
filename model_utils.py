from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import metrics
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

from Utils import countries_dict

country_dict_all = countries_dict.copy()
country_dict_all['other'] = 'Other Countries'
country_dict_all['NDF'] = 'No destination found'
country_dict_all['all'] = 'ALL'

# cols = ['gender', 'signup_method', 'signup_flow', 'language',
#         'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
#         'signup_app', 'first_device_type', 'first_browser',
#         ]

cols = ['gender', 'signup_method', 'language', 'first_affiliate_tracked',
        'affiliate_channel','signup_app', 'affiliate_provider',
        'first_device_type', 'first_browser',
        ]

AGE_method ={
    'Imputation by mean':'mean',
    'Imputation by median': 'median',
    'Imputation by backfill': 'bfill',
    'Imputation by forwardfill': 'ffill',
}

FAT_method= {
    'Imputation by mode':'mode',
    'Imputation by random value':'random',
    'Imputation by backfill':'bfill',
    'Imputation by forwardfill':'ffill',
}
def fill_missing_numerical(df, method, IQR_ratio=5, show_outlayers_num=False, APPLY_ZSCORE = False,
                           show_dist_plot=False, show_values_range=False, Outlayers_remove_IQR= False):
       
    #* getting rid of wrong inputs

    for values in df[df > 999].unique():
        df.replace(values, 2015-values, inplace= True)
    #* filling missing values
    
    if method == 'median':
        df = df.replace(np.NaN, df.median())
    elif method == 'mean':
        df = df.replace(np.NaN, round(df.mean(), 0))
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'ffill':
        df = df.fillna(method='ffill')
    # df = df.replace(np.nan, df.median()).reset_index(drop=True)
    origional_len = df.isna().value_counts()
    # print(origional_len)
    
    if Outlayers_remove_IQR :
        df.describe()
        Q1 = df.describe().loc['25%']
        Q3 = df.describe().loc['75%']
        IQR = Q3 - Q1
        min_range = Q1 - IQR_ratio*IQR
        max_range = Q3 + IQR_ratio*IQR
        if show_values_range:
            print(f'[min range: {min_range}, -  max range: {max_range} ]')
            
        df_imputed = df[(df < max_range) & (df > min_range)]
        counts = df_imputed.count()
        if show_outlayers_num:
            print(f"Outlayers removed : {origional_len[0] - counts }")

    else :
        df_imputed = df

    if show_dist_plot:
        df_imputed.plot(kind='hist', bins=120)
    if APPLY_ZSCORE:
        # df_imputed = zscore(df_imputed)
        from scipy import stats
        df_imputed = df_imputed[(np.abs(stats.zscore(df_imputed)) < 3).all(axis=1)]
    return df_imputed


def age_cleaner(age, ratio):
    age.describe()
    Q1 = age.describe().loc['25%']
    Q3 = age.describe().loc['75%']
    IQR = Q3 - Q1
    min_range = Q1 - ratio*IQR
    max_range = Q3 + ratio*IQR
    return min_range, max_range

def fill_missing_categorical(df, method):
    if method =='mode':
        df = df.replace(np.NaN, df.mode()[0])
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'random':
        import random
        vals = list(df.dropna().unique())
        df = df.replace(np.NaN, random.choice(vals))
    return df


def distribution_plot_numerical(df, title='Age', feature2_val='all', bins=120, feature1='age', feature2='country_destination'):
    fig = make_subplots(rows=3, cols=2, x_title=title, y_title='Count', row_heights=[0.4, 0.4, 0.4],
                        column_widths=[0.4, 0.4], vertical_spacing=0.1, horizontal_spacing=0.05,
                        subplot_titles=['Orginial Distribution', 'Median-fixed Distribution',
                                        'Mean-fixed Distribution', 'Back-filled Distribution', 'Forward-filled Distribution'],
                        specs=[[{"colspan": 2}, None],
                               [{}, {}],
                               [{}, {}],
                               ])

    if feature2_val == 'all':
        df = df[feature1]
    else:
        df = df[df[feature2] == feature2_val][feature1]

    x1 = df
    median_fixed = fill_missing_numerical(x1, 'median', 5)
    mean_fixed = fill_missing_numerical(x1, 'mean', 5)
    back_filled = fill_missing_numerical(x1, 'bfill', 5)
    forward_filled = fill_missing_numerical(x1, 'ffill', 5)
    min_range, max_range = age_cleaner(x1, 5)
    x1 = x1[(x1 < max_range) & (x1 > min_range)]

    Original_dist = go.Histogram(
        x=x1, nbinsx=bins, name="Orginial")

    median_fixed_dist = go.Histogram(
        x=median_fixed,  nbinsx=bins, name="Median-fixed")
    mean_fixed_dist = go.Histogram(
        x=mean_fixed,  nbinsx=bins, name="Mean-fixed")
    back_filled_fixed_dist = go.Histogram(
        x=back_filled,  nbinsx=bins, name="Back-filled")
    forward_filled_fixed_dist = go.Histogram(
        x=forward_filled,  nbinsx=bins, name="Forward-filled")

    fig.append_trace(Original_dist, 1, 1)
    fig.append_trace(median_fixed_dist, 2, 1,)

    # fig.append_trace(Original_dist, 1, 2)
    fig.append_trace(mean_fixed_dist, 2, 2)

    # fig.append_trace(Original_dist, 2, 1)
    fig.append_trace(back_filled_fixed_dist, 3, 1)

    # fig.append_trace(Original_dist, 2, 2)
    fig.append_trace(forward_filled_fixed_dist, 3, 2)
    # fig.update_layout(title_text="Age Distribution", xaxis_title="Age", yaxis_title="Count")
    fig.update_traces(opacity=0.65)
    fig.update_layout(height=600)
    if feature2_val == 'all':
        fig.update_layout(
            title_text=f'{title} distributions of All countries', title_x=0.3)
    else:
        fig.update_layout(
            title_text=f'{title} distributions of {country_dict_all[feature2_val]}', title_x=0.3)
    fig.update_layout(template='plotly_dark')
    fig.update_layout(font_size=15)
    fig.update_layout(title_font_size=24)
    fig.update_xaxes(rangeselector_font_size=20)
    # fig.update_xaxes(tickfont_size=20)
    fig.update_traces(legendgrouptitle_font_size=25,
                      selector=dict(type='histogram'))
    st.plotly_chart(fig, use_container_width=True)




def distribution_plot_categorical(df, title='FAT', feature2_val='all', bins=12, feature1 = 'first_affiliate_tracked', feature2 ='country_destination'):

    fig = make_subplots(rows=3, cols=2,  y_title='Count', row_heights=[0.4, 0.4, 0.4],
                        column_widths=[0.4, 0.4], vertical_spacing=0.23, horizontal_spacing=0.05,
                        subplot_titles=['Orginial', 'Mode-fixed Distribution',
                                        'Random-fixed Distribution', 'Back-filled Distribution', 'Forward-filled Distribution'],
                        specs=[[{"colspan": 2}, None],
                               [{}, {}],
                               [{}, {}],
                               ])

    if feature2_val == 'all':
        df = df[feature1]
    else:
        df = df[df[feature2] == feature2_val][feature1]

    mode_fixed = fill_missing_categorical(df, 'mode')
    random_fixed = fill_missing_categorical(df, 'random')
    back_filled = fill_missing_categorical(df, 'bfill')
    forward_filled = fill_missing_categorical(df, 'ffill')



    Original_dist = go.Histogram(
        x=df, nbinsx=bins, name="Orginial")

    median_fixed_dist = go.Histogram(
        x=mode_fixed,  nbinsx=bins, name="Mode-fixed")
    mean_fixed_dist = go.Histogram(
        x=random_fixed,  nbinsx=bins, name="Random-fixed")
    back_filled_fixed_dist = go.Histogram(
        x=back_filled,  nbinsx=bins, name="Back-filled")
    forward_filled_fixed_dist = go.Histogram(
        x=forward_filled,  nbinsx=bins, name="Forward-filled")

    fig.append_trace(Original_dist, 1, 1)
    fig.append_trace(median_fixed_dist, 2, 1,)

    # fig.append_trace(Original_dist, 1, 2)
    fig.append_trace(mean_fixed_dist, 2, 2)

    # fig.append_trace(Original_dist, 2, 1)
    fig.append_trace(back_filled_fixed_dist, 3, 1)

    # fig.append_trace(Original_dist, 2, 2)
    fig.append_trace(forward_filled_fixed_dist, 3, 2)
    # fig.update_layout(title_text="Age Distribution", xaxis_title="Age", yaxis_title="Count")
    fig.update_traces(opacity=0.65)
    fig.update_layout(height=600)
    if feature2_val == 'all':
        fig.update_layout(
            title_text=f'{title} distributions of all countries', title_x=0.4)
    else:
        fig.update_layout(
            title_text=f'{title} distributions of {country_dict_all[feature2_val]}', title_x=0.4)
    fig.update_layout(template='plotly_dark')
    #fig.update_layout(font_size=15)
    fig.update_layout(title_font_size=24)
    fig.update_xaxes(rangeselector_font_size=20)
    # fig.update_xaxes(tickfont_size=20)
    fig.update_traces(legendgrouptitle_font_size=25,
                      selector=dict(type='histogram'))
    # fig.update_layout(margin_pad=10)
    st.plotly_chart(fig, use_container_width=True)



def discrete_categories(df, col):
    # testing = df.drop(['id', 'date_account_created',
    #                   'timestamp_first_active', 'date_first_booking',], axis=1, )
    testing = df
    #testing['gender'] = testing['gender'].astype(str)
    
    for _, value in enumerate(col):
        if len(testing[value].unique()) >= 0:
            testing[value] = testing[value].replace(
                list(testing[value].unique()), range(0, len(list(testing[value].unique()))))
    testing["age"] = testing["age"].replace(np.nan, 0)
    return testing

def print_score(clf, xtrain, ytrain, xtest, ytest):
    clf = clf.fit(xtrain, ytrain)
    y_pred = clf.predict(xtest)
    return metrics.accuracy_score(ytest, y_pred)

def unbaised_sample(df, random=12):

    train_cleaned = df.copy()
    #train_cleaned = train_cleaned.drop(['timestamp_first_active','date_account_created','date_first_booking'], axis=1)
    train_sized = train_cleaned.groupby('country_destination').size().reset_index()
    train_sized.rename(columns={0: 'count'}, inplace=True)
    train_sized['count'] = (train_sized['count']/10).apply(np.floor)

    train_sized
    count = 0
    for idx, value in enumerate(train_sized['country_destination']):
        sample = shuffle(train_cleaned[train_cleaned['country_destination']
                         == value], random_state=random)
        sample = sample[0:int(train_sized.iloc[idx][1])]
        if count == 0:
            test_df = sample
            count = 1

        else:
            test_df = pd.concat([test_df, sample])
            # print("entered")

    return test_df

def Time_series_forcast(df , train_y, test_y, predictions, future_forcast ,year = 2010):
    # year = 2010
    counts = df
    year2010 = counts[(counts['Date'] > str(year))& (counts['Date'] < str(year+1))].reset_index() ; year += 1
    year2011 = counts[(counts['Date'] > str(year))& (counts['Date'] < str(year+1))].reset_index() ; year += 1
    year2012 = counts[(counts['Date'] > str(year))& (counts['Date'] < str(year+1))].reset_index() ; year += 1
    year2013 = counts[(counts['Date'] > str(year))& (counts['Date'] < str(year+1))].reset_index() ; year += 1
    year2014 = counts[(counts['Date'] > str(year))& (counts['Date'] < str(year+1))].reset_index() ; year += 1


    fig = go.Figure()

    YEAR_SPAN = True

    if YEAR_SPAN:
        #* MOnths+Days
        Line_year2010 = go.Scatter(x=year2010.index, y=year2010['count'], name="Year 2010")
        Line_year2011 = go.Scatter(x=year2011.index, y=year2011['count'], name="Year 2011")
        Line_year2012 = go.Scatter(x=year2012.index, y=year2012['count'], name="Year 2012")
        Line_year2013 = go.Scatter(x=year2013.index, y=year2013['count'], name="Year 2013")
        #Line_year2014 = go.Scatter(x=year2014.index, y=year2014['count'], name="Year 2014")
        Line_year2014_to_may = go.Scatter(x=train_y.reset_index().index, y=train_y.values, name="Year 2014_train")
        Line_year2014_June_real = go.Scatter(x = predictions.index, y = test_y.values, name="Year 2014_test")
        Line_year2014_June_predict = go.Scatter(x = predictions.index, y = predictions.values.astype(int), name="Year 2014_predict")
        Line_year2014_July_forecast = go.Scatter(x = future_forcast.index, y = future_forcast.values.astype(int), name="Year 2014_July_forecast")

        
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = [1,30,62,90,120,150,180,211,242,273,304,335],
                ticktext = ['Januray','February','March','April','May','June','July','August','September','October','November','December'],
                tickangle = 30
            )
        )
        fig.update_layout(xaxis_title="Months - Span")
    else :
    #* Years
        Line_year2010 = go.Scatter(x=year2010['Date'], y=year2010['count'], name="Year 2010")
        Line_year2011 = go.Scatter(x=year2011['Date'], y=year2011['count'], name="Year 2011")
        Line_year2012 = go.Scatter(x=year2012['Date'], y=year2012['count'], name="Year 2012")
        Line_year2013 = go.Scatter(x=year2013['Date'], y=year2013['count'], name="Year 2013")
        Line_year2014 = go.Scatter(x=year2014['Date'], y=year2014['count'], name="Year 2014")
        
        
        
        fig.update_xaxes(nticks=15)
        fig.update_layout(xaxis_title="Years - Span")


    fig.add_trace(Line_year2010)
    fig.add_trace(Line_year2011)
    fig.add_trace(Line_year2012)
    fig.add_trace(Line_year2013)
    #fig.add_trace(Line_year2014)
    fig.add_trace(Line_year2014_to_may)
    fig.add_trace(Line_year2014_June_real)
    fig.add_trace(Line_year2014_June_predict)
    fig.add_trace(Line_year2014_July_forecast)
        
        
    fig.update_layout(
            title="Number of Registers per day- from 2010 to 2014",
            xaxis_title="Month - Span",
            yaxis_title="Counts",
            title_x=0.5,
            template='plotly_dark'
    )
    st.plotly_chart(fig)
