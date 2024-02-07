import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st 
import matplotlib.pyplot as plt

model = load_model(r'D:\NEW PROJECT SMP\modelstock.h5')

st.header('Stock Market Predictor Using ML')

stock= st.text_input('Enter stock symbol here: ', 'GOOG')
start='2012-01-01'
end='2022-12-31'



data=yf.download(stock, start, end)

st.subheader('Stock DataSet')

st.write(data) 

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)]) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True) 
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs avg 100 days data')
ma_100_days = data['Close'].rolling(100).mean()
fig1 = plt.figure(figsize=(6, 4))
plt.plot(ma_100_days, 'pink', label='MA (100 days)')
plt.plot(data['Close'], 'lightblue', label='Close Price')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs avg 50 and avg 100 days data')
ma_50_days = data['Close'].rolling(50).mean()
fig2 = plt.figure(figsize=(6, 4))
plt.plot(ma_50_days, 'brown', label='MA (50 days)')
plt.plot(ma_100_days, 'pink', label='MA (100 days)')
plt.plot(data['Close'], 'lightblue', label='Close Price')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs avg 100 vs avg 200 data')
ma_200_days = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(6, 4))
plt.plot(ma_200_days, 'brown', label='MA (200 days)')
plt.plot(ma_100_days, 'pink', label='MA (100 days)')
plt.plot(data['Close'], 'lightblue', label='Close Price')
plt.show()
st.pyplot(fig3)

x=[]
y=[]

for i in range (100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale 
y = y * scale 

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(6, 4))
plt.plot(predict, 'lightgreen', label='Original Price')
plt.plot(y, 'lightblue', label='Predicted Price')
plt.xlabel('TIME')
plt.ylabel('PRICE')
plt.show()
st.pyplot(fig4)