import streamlit as st
from datetime import date
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import datetime
import matplotlib.ticker as ticker
from plotly import graph_objs as go
from keras.models import load_model
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT','SBIN.NS','TSLA','GC=F','Other')
symbol = st.selectbox('Select dataset for prediction', stocks)
if symbol == 'Other':
    symbol = st.text_input('Company stock Ticker','MU')

@st.cache_data
def load_data(symbol):
    url = f'https://finance.yahoo.com/quote/{symbol}/history'
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.43'}
    response = requests.get(url, headers = headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    data = []
    table = soup.find('table', {'class': 'W(100%) M(0)'})
    if table:
        rows = table.find_all('tr', {'class': 'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)'})
        if rows:
            for row in rows:
                date = row.find_all('td')[0].text
                open_price = row.find_all('td')[1].text
                # Check if open_price contains 'dividend'
                if 'dividend' in open_price.lower():
                    # If 'dividend' is present, mark all corresponding values as 'NA'
                    open_price, high_price, low_price, close_price, adj_close_price, volume = 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
                else:
                    # Extract other values if open_price is not 'dividend'
                    high_price = row.find_all('td')[2].text
                    low_price = row.find_all('td')[3].text
                    close_price = row.find_all('td')[4].text
                    adj_close_price = row.find_all('td')[5].text
                    volume = row.find_all('td')[6].text

                    # Append the extracted data to the list as a dictionary
                    data.append({'date':date,
                                 'Open': open_price,
                                 'High': high_price,
                                 'Low': low_price,
                                 'Close': close_price,
                                 'Adj Close': adj_close_price,
                                 'Volume': volume})
    df = pd.DataFrame(data)
    
    def str_to_datetime(s):
        split = s.split(' ')
        month,day,year = split[0],split[1],int(split[2])
        d = day.split(',')
        day = int(d[0])
        dic = {'Jun':6,'Jan':1,'Dec':12,'May':5,'Jul':7,'Feb':2,'Apr':4,'Mar':3,'Aug':8,'Sep':9,'Nov':11,'Oct':10}
        return datetime.datetime(year = year, month = dic[month], day = day)

    df['date'] = df['date'].apply(str_to_datetime)
    df.index = df.pop('date')
    
    df['Close'] = df['Close'].str.replace(',', '')
    df['Close'] = pd.to_numeric(df['Close'])
    df = df.dropna(subset=['Close'])
        
    data_table = df
    data_table.reset_index(inplace=True)
    return data_table


data_load_state = st.text('Loading data...')
data = load_data(symbol)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.head())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['date'], y=data['Close'], name="stock_close"))
    fig.add_trace(go.Scatter(x=data['date'], y=data['High'], name="stock_high"))
    fig.add_trace(go.Scatter(x=data['date'], y=data['Low'], name="stock_low"))
    
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
    st.subheader('Closing Price vs Time chart')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data['Close'])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45)
    plt.ylabel('Closing Price')
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 3-day Moving Average')
    ma6 = data['Close'].rolling(3).mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ma6, 'r', label='3-day Moving Average')
    ax.plot(data['Close'], label='Closing Price')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45)
    plt.ylabel('Closing Price')
    plt.legend()
    st.pyplot(fig)

plot_raw_data()
    
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70) : int(len(data))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('C:/Users/Smilika/Documents/ML Mini project/keras_model.h5')


past_4_days = data_training.tail(4) 
final_df = pd.concat([past_4_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_train_val = []
y_train_val = []

for i in range(4,input_data.shape[0]):
    x_train_val.append(input_data[i-4:i])
    y_train_val.append(input_data[i,0])
    
x_train_val,y_train_val = np.array(x_train_val), np.array(y_train_val)

y_predicted = model.predict(x_train_val)
scalers = scaler.scale_
scale_fact = 1/scalers[0]
y_predicted = y_predicted *scale_fact
y_train_val = y_train_val*scale_fact


st.subheader('Predictionvs Original')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_train_val,'b',label = 'ori price')
plt.plot(y_predicted,'r',label = 'pred price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)