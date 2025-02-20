import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download Tesla stock data
# tsla = yf.download("TSLA", start="2024-02-19", end="2025-02-19")

# # 只保留需要的列，并重命名
# tsla = tsla[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
# tsla.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# # 保存为 CSV 文件
# tsla.to_csv("Tesla_Stock_Data.csv", index=False)

# print("Tesla 股票数据已保存为 Tesla_Stock_Data.csv")

#RNN_predict_stock_price.py
data = pd.read_csv("./data/Tesla_Stock_Data.csv")
#print(data.head())
price = data['Close']
#print(price.head())
#Normalize the data
price_norm = price/max(price)
#print(price_norm)

#Display stock data
fig1= plt.figure(figsize=(10,5))
plt.plot(price)
plt.title("Tesla Stock Price")
plt.xlabel("Days")
plt.ylabel("Price")
#plt.show()

#Extract the training data
def extract_data(data, time_step):
    X, y = [], []
    #0,1,2,3,4,5,6,7,8,9:10 samples, time_step=8;Use the first 8 samples to predict the 9th sample
    for i in range(len(data)-time_step):
        X.append([a for a in data[i:i+time_step]])
        y.append(data[i+time_step])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, np.array(y)

#Define X and y
time_step=8
X,y = extract_data(price_norm, time_step)

print("Number of samples (X):", X.shape[0])
print("Number of labels (y):", len(y))
#Set up the RNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense,Input
model = Sequential()

model.add(Input(shape=(time_step, 1)))
model.add(SimpleRNN(units=5, activation="relu"))
model.add(Dense(units=1,activation="linear"))
model.compile(optimizer="adam", loss="mean_squared_error")
#model.summary()

model.fit(X, y, epochs=200, batch_size=22)

#Predict the stock price
#Convert the data to the same format as the training data
y_train_pred = model.predict(X)*max(price)
y_train = [i*max(price) for i in y]
fig2 = plt.figure(figsize=(10,5))
plt.plot(y_train,labels="Real Price")
plt.plot(y_train_pred,labels="Predicted Price")
plt.title("Tesla Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

#预测模型存在延时