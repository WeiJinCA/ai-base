

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data  = pd.read_csv('./data/Binary_Classification_400_Samples.csv')
X= data.drop(['y'],axis=1)
y = data.loc[:,'y']
print(X.head())
print(y.head())

#split the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=10, stratify=y)
# print(X_train.shape, X_test.shape, y_train.shape, X.shape)
# print(y_train)
# print(pd.Series(y_train).value_counts())
# print(pd.Series(y_test).value_counts())

# #visualize the data
# fig1= plt.figure(figsize=(5,5))
# passed = plt.scatter(X.loc[:,'x1'][y==1], X.loc[:,'x2'][y==1])
# failed = plt.scatter(X.loc[:,'x1'][y==0], X.loc[:,'x2'][y==0])

# plt.legend((passed, failed), ('passed', 'failed'))
# plt.xlabel('x1')
# plt.ylabel('x2')

# # Create a Sequential model
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()

# # Add layers to the model
model.add(Dense(units = 20, input_dim=2, activation='sigmoid'))
model.add(Dense(units = 1, activation='sigmoid'))

# #check the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
model.fit(X_train, y_train, epochs=4000)

# make predictions and calculate accuracy
y_train_pred = (model.predict(X_train) > 0.5).astype("int32")
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train, y_train_pred)
print('Accuracy: %.2f' % (accuracy*100))

#make predictions on the test data
y_test_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy_test = accuracy_score(y_test, y_test_pred)
print('Accuracy: %.2f' % (accuracy_test*100))

#visualize the decision boundary
y_train_pred_format = pd.Series(y_train_pred.ravel())

#generate new data for plot
XX,yy= np.meshgrid(np.arange(0,1,0.01), np.arange(0,1,0.01))
X_range = np.c_[XX.ravel(), yy.ravel()]
y_range_predict = (model.predict(X_range) > 0.5).astype("int32")
y_range_train_pred_format = pd.Series(y_range_predict.ravel())

#plot the decision boundary
fig2= plt.figure(figsize=(5,5))
passed_predict = plt.scatter(X_range[:,0][y_range_train_pred_format==1], X_range[:,1][y_range_train_pred_format==1])
failed_predict = plt.scatter(X_range[:,0][y_range_train_pred_format==0], X_range[:,1][y_range_train_pred_format==0])

passed = plt.scatter(X.loc[:,'x1'][y==1], X.loc[:,'x2'][y==1])
failed = plt.scatter(X.loc[:,'x1'][y==0], X.loc[:,'x2'][y==0])
# plt.legend((passed, failed,passed_predict,failed_predict), ('passed', 'failed','passed_predict','failed_predict'))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Prediction result')
plt.show()
