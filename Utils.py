import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st


age_dict = {
    '100+': 20,
    '95-99': 19,
    '90-94': 18,
    '85-89': 17,
    '80-84': 16,
    '75-79': 15,
    '70-74': 14,
    '65-69': 13,
    '60-64': 12,
    '55-59': 11,
    '50-54': 10,
    '45-49': 9,
    '40-44': 8,
    '35-39': 7,
    '30-34': 6,
    '25-29': 5,
    '20-24': 4,
    '15-19': 3,
    '10-14': 2,
    '5-9': 1,
    '0-4': 0
}

countries_dict = {
    'PT': 'Portugal',
    'NL' : 'Netherlands',
    'AU' : 'Australia',
    'CA' : 'Canada',
    'ES' : 'Spain',
    'IT' : 'Italy',
    'GB' : 'United Kingdom',
    'FR' : 'France',
    'DE' : 'Germany',
    'US' : 'United States',
}

def bar_coloring(bar, ax):
    bar_color = bar[0].get_facecolor()
    height = (bar[0].get_height()/100)*2
    for bar in bar:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + height,
            int(round(bar.get_height(), 0)),
            horizontalalignment='center',
            color=bar_color,
            fontsize=7,
            weight='bold'
        )

def dejunking(ax,fig):
        
    x = plt.gca().xaxis
    for item in x.get_ticklabels():
        item.set_fontsize(7.5)
        item.set_rotation(45)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.tick_params(bottom=False, left=False)

    ax.set(facecolor='lightgray')
    fig.set(facecolor='lightgray')

    plt.grid(axis='x', visible=False)
    ax.get_yaxis().set_ticks([])
    fig.tight_layout()
    ax.margins(x=0.01)
    ax.margins(y=0.005)

    plt.legend(loc ='best')
    

def bar_plot(df, choosen_gender, destination):
    if choosen_gender == 'both':
        cond2 = df['country_destination'] == destination
        df1 = df[cond2]
        width = 0.85

    else:
        cond1 = df['gender'] == choosen_gender
        cond2 = df['country_destination'] == destination
        df1 = df[cond1 & cond2]
        width = 1.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = 0
    for element in df1.gender.unique():
        if element == 'male':
            color = '#1f77b4'
        elif element == 'female':
            color = '#d62728'
        bars = plt.bar(
            x=(df1[df1['gender'] == element]['values']+bar_width)*2,
            height=df1[df1['gender'] == element]['population_in_thousands'],
            label=element,
            width=width,
            color=color,
        )
        bar_width = 0.5
        bar_coloring(bars, ax)

    dejunking(ax, fig)
    if choosen_gender == 'both':
        plt.xticks(df1['values']*2+0.5, df1['age_bucket'])
    else:
        plt.xticks(df1['values']*2+0.5, df1['age_bucket'])
    plt.legend(loc='best')

    plt.title("Flights of {}s to {}".format(
        choosen_gender, countries_dict[destination]), fontsize=23, pad=35)
    plt.xlabel('Age Bucket', fontsize=17,)
    plt.ylabel('Population in Thousands', fontsize=17,)
    # plt.show()
    st.pyplot(fig)
    

def box_plot(df, choosen_gender, destination, orientation='v'):
    if orientation == 'v':
        x_axies = {'gender' : "Gender"}
        y_axies = {'population_in_thousands' : "Population in Thousands"}
    elif orientation == 'h':
        y_axies = {'gender' : "Gender"}
        x_axies = {'population_in_thousands' : "Population in Thousands"}

    theme = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    if choosen_gender == 'both':
        cond2 = df['country_destination'] == destination
        df1 = df[cond2]

    else:
        cond1 = df['gender'] == choosen_gender
        cond2 = df['country_destination'] == destination
        df1 = df[cond1 & cond2]

        
    fig = px.box(df1, x =list(x_axies.keys())[0], y = list(y_axies.keys())[0], points= 'all', 
                color = 'gender',template= theme[2], hover_data=['age_bucket'  , 'population_in_thousands'],
                labels = {'population_in_thousands': 'Population ', 'age_bucket': 'Age ',
                        'gender': 'Gender '},color_discrete_map={ "male": "#1f77b4", "female": "#d62728"
                },
                notched=False,)
             

    fig.update_layout(height=600, width=800)
    fig.update_layout(title_text='Flights of {}s to {}'.format(choosen_gender, destination), title_x=0.5)
    # fig.show()
    st.plotly_chart(fig, use_container_width=True)




def heatmap_plottly(df):
    theme = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
    fig = px.imshow(df.corr().round(3), text_auto="True", template=theme[2],
                    color_continuous_scale="ylgnbu", aspect='auto', range_color=[-1, 1])


    fig.update_layout(height=600, width=800)
    fig.update_yaxes(tickangle=300, tickfont=dict(size=18))
    fig.update_xaxes(tickfont=dict(size=18))
    fig.update_layout(margin_pad=10)
    fig.update_layout(font_size=18)
    fig.update_layout(title_text='Correlation between columns', title_x=0.5)
    st.plotly_chart(fig, use_container_width=True)
    
    