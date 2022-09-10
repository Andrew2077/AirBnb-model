from Utils import*
from model_utils import*
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
import warnings
import datetime
warnings.filterwarnings('ignore')


train = pd.read_csv('airbnb/train_users_2.csv')

st.markdown("""
## 📈 $\color{#69109c}{\t{Time series - SKforecast}}$ 
Time series forcasting using SKforecast
let's get the Df ready for the model

- create a new df with the date and the number of users
```python
Dates = train.groupby('date_account_created')[['id']].count()
counts = Dates.copy()
``` 
            """)
Dates = train.groupby('date_account_created')[['id']].count()
counts = Dates.copy()

st.markdown("""
- adjust the index and time series column
```python
counts = counts.reset_index()
counts.rename(columns={'id':'count','date_account_created':'Date'}, inplace=True)
counts['Date'] =pd.to_datetime(counts['Date'])
```
""")
counts = counts.reset_index()
counts.rename(columns={'id':'count','date_account_created':'Date'}, inplace=True)
counts['Date'] =pd.to_datetime(counts['Date'])

st.markdown("""
- Creating a period time series for forecasting
```python
start = datetime.datetime(2014, 7, 1)
end = datetime.datetime(2014, 7, 31)
index = pd.date_range(start, end)
```
- merge it with the previous df

```python
forecast =pd.DataFrame(index, columns=['Date'])
counts = pd.merge(counts, forecast, on='Date', how='outer')
counts.replace(np.NaN, 0, inplace=True)
```

""")
start = datetime.datetime(2014, 7, 1)
end = datetime.datetime(2014, 7, 31)
index = pd.date_range(start, end)

forecast =pd.DataFrame(index, columns=['Date'])
counts = pd.merge(counts, forecast, on='Date', how='outer')
counts.replace(np.NaN, 0, inplace=True)

st.markdown("""
            
- Getting Data ready for the model
```python
y = counts[(counts['Date']>='2014-01-01') & (counts['Date']<'2014-07-01')].set_index('Date')['count']

to_be_predicted = counts[(counts['Date']>='2014-07-01')].set_index('Date')['count'].astype(int)
to_be_predicted = to_be_predicted.reset_index()
to_be_predicted.index = range(182,213,1)
train_y= y.iloc[:151].copy()
test_y = y.iloc[151:].copy()
```
---
---
""")

y = counts[(counts['Date']>='2014-01-01') & (counts['Date']<'2014-07-01')].set_index('Date')['count']

to_be_predicted = counts[(counts['Date']>='2014-07-01')].set_index('Date')['count'].astype(int)
to_be_predicted = to_be_predicted.reset_index()
to_be_predicted.index = range(182,213,1)
train_y= y.iloc[:151].copy()
test_y = y.iloc[151:].copy()

st.markdown("""
## $\color{#69109c}{\t{AutoReg - RandomForest}}$ 

### Training and testing the model

```python
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor


forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=25, n_estimators=200, max_depth=5),
                lags = 70 # more lages, enforces more stationarity
                )
```
- Fitting & predicting
```python
forecaster.fit(y=train_y)
steps = len(test_y)
predictions = forecaster.predict(steps=steps)

```
- Evaluating the model, MSE

```python
error_mse = mean_squared_error(
                y_true = test_y,
                y_pred = predictions
            )
            
print(f"Test error (mse): {error_mse}")
```
""")


forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=25, n_estimators=200, max_depth=5),
                lags = 70
                )

forecaster.fit(y=train_y)

steps = len(test_y)
predictions = forecaster.predict(steps=steps)

error_mse = mean_squared_error(
                y_true = test_y,
                y_pred = predictions
            )

forecaster.fit(y=y)
steps2 = len(to_be_predicted)
future_forcast = forecaster.predict(steps=steps2)

st.write(f"Test error (mse) - Intial : 16458.755351612905")
st.write(f"Test error (mse) - adjusted : {error_mse}")

st.markdown("""
- predicting the next 30 days 
```python
forecaster.fit(y=y)
steps2 = len(to_be_predicted)
future_forcast = forecaster.predict(steps=steps2)
```

""")

st.markdown("""
## $\color{#69109c}{\t{Forcast - Plot}}$ 
""")

Time_series_forcast(counts , train_y, test_y, predictions, future_forcast ,year = 2010)
