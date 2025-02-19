import numpy as np
import pandas as pd
from keras.datasets import mnist
from matplotlib import pyplot as plt


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(type(X_train),X_train.shape) #60000个28*28的2D array

#visualize the img example
img1= X_train[0]
# fig1= plt.figure(figsize=(3,3))
# plt.imshow(img1)
# plt.title(y_train[0])
# plt.show()

#reshape the data
feature_size = img1.shape[0]*img1.shape[1]
X_train_format = X_train.reshape(X_train.shape[0], feature_size)
X_test_format = X_test.reshape(X_test.shape[0], feature_size)
print(X_train_format.shape) #60000个784的1D array

#normalize the data
X_train_normal = X_train_format/255
X_test_normal = X_test_format/255

#format the output data(labels)
from keras.utils import to_categorical
y_train_format = to_categorical(y_train)
y_test_format = to_categorical(y_test)
#print(y_train_format[0]) 


# # Create a Sequential model
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()

# # # Add layers to the model
model.add(Dense(units = 392, input_dim=feature_size, activation='sigmoid'))
model.add(Dense(units = 392,  activation='sigmoid'))
model.add(Dense(units = 10, activation='softmax'))

# #check the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(X_train_normal, y_train_format, epochs=10)

# make predictions and calculate accuracy
y_train_pred = np.argmax(model.predict(X_train_normal), axis=1)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train, y_train_pred)
print('Accuracy: %.2f' % (accuracy*100))

#make predictions on the test data
y_test_pred = np.argmax(model.predict(X_test_normal), axis=1)
accuracy_test = accuracy_score(y_test, y_test_pred)
print('Accuracy: %.2f' % (accuracy_test*100))

#test the model by showing the img
img2= X_test[10]
fig2= plt.figure(figsize=(3,3))
plt.imshow(img2)
plt.title(y_test_pred[10]) #show the predicted label
plt.show()
