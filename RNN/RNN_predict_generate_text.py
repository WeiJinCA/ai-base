import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#RNN_predict_stock_price.py
data = open("./data/RNN_repeated_text.txt").read()
data = data.replace("\n", " ")
data = data.replace("\r", " ")
#print(data)

#Dlete the repeated words
letters = list(set(data))
num_letters = len(letters)
#print("Number of letters:", num_letters)

#Create a dictionary to map the letters to numbers
int_to_char = {a:b for a,b in enumerate(letters)}
char_to_int = {b:a for a,b in enumerate(letters)}
# print(int_to_char)
# print(char_to_int)

time_step=20

#Data processing
from keras.utils import to_categorical
#Batch conversion from chat to int
def extract_data(data, slide):
    X, y = [], []
    for i in range(len(data)-slide):
        X.append([a for a in data[i:i+slide]])
        y.append(data[i+slide])

    return X, y

#Batch conversion from chat to int
def char_to_int_Data(x,y,chat_to_int):
    x_to_int, y_to_int = [], []
    for i in range(len(x)):
        x_to_int.append([char_to_int[a] for a in x[i]])
        y_to_int.append([char_to_int[y[i]]])

    return x_to_int, y_to_int

#Batch conversion based on char article input, 
def data_processing(data, slide,letters,char_to_int):
    char_Data = extract_data(data, slide)

    int_Data = char_to_int_Data(char_Data[0], char_Data[1], char_to_int)
    Input = int_Data[0]
    Output = list(np.array(int_Data[1]).flatten())
    Input_RESHAPED = np.array(Input).reshape(len(Input), slide)
    new = np.random.randint(0, 10,size=[Input_RESHAPED.shape[0], Input_RESHAPED.shape[1],num_letters])
    for i in range(Input_RESHAPED.shape[0]):
        for j in range(Input_RESHAPED.shape[1]):
            new[i,j,:] = to_categorical(Input_RESHAPED[i,j], num_classes=num_letters)

    return new, Output

#Extract X and y
X, y = data_processing(data, time_step, num_letters, char_to_int)
print(X.shape)

#Split the data into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=10)
print(X_train.shape)

y_train_category = to_categorical(y_train, num_classes=num_letters)
y_test_category = to_categorical(y_test, num_classes=num_letters)

#Set up the RNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
model = Sequential()

model.add(LSTM(units=20, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=num_letters,activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=['accuracy']) #multi-class classification
model.summary()

model.fit(X_train, y_train_category, epochs=16, batch_size=1000)

#Predict
y_train_pred = model.predict(X_train)
y_train_pred = np.argmax(y_train_pred, axis=1)
y_train_pred_char = [int_to_char[i] for i in y_train_pred]
print("Predicted text:", "".join(y_train_pred_char))

#Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score_train = accuracy_score(y_train, y_train_pred)
print("Accuracy:", accuracy_score_train)

#Evaluate the model
y_test_pred = model.predict(X_test)
y_test_pred = np.argmax(y_test_pred, axis=1)
accuracy_score_test = accuracy_score(y_test, y_test_pred)
print("Accuracy Test:", accuracy_score_test)

#Predict the text
new_text = "flare is a teacher in ai industry. He obtained his phd in Australia."
X_new, y_new = data_processing(new_text, time_step, num_letters, char_to_int)
y_new_pred = model.predict(X_new)
y_new_pred = np.argmax(y_new_pred, axis=1)
y_new_pred_char = [int_to_char[i] for i in y_new_pred]
print("Predicted text:", "".join(y_new_pred_char))