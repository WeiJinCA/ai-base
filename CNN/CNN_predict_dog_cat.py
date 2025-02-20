#Load data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('./data/cnn_dog_cat/training_set',
                                                 target_size=(50, 50),
                                                 batch_size=32,
                                                 class_mode='binary')

#Set up the CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(50, 50, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#FC layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#Train the model
model.fit(training_set, epochs=18)

#Accuracy on training set
accuracy_train = model.evaluate(training_set)
print("accuracy_train",accuracy_train)

#Accuracy on test set
test_set = train_datagen.flow_from_directory('./data/cnn_dog_cat/test_set',
                                                 target_size=(50, 50),
                                                 batch_size=32,
                                                 class_mode='binary')
accuracy_test = model.evaluate(test_set)
print("accuracy_test",accuracy_test)

#Predict
#Load single image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
image = load_img('./data/cnn_dog_cat/dog.jpeg', target_size=(50, 50))
image = img_to_array(image)
image = image/255
image = image.reshape(1, 50, 50, 3)
prediction = model.predict(image)
print(prediction)
if prediction > 0.5:
    print("Dog")
else:
    print("Cat")

img_cat = load_img('./data/cnn_dog_cat/cat.jpeg', target_size=(50, 50))
img_cat = img_to_array(img_cat)
img_cat = img_cat/255
img_cat = img_cat.reshape(1, 50, 50, 3)
prediction1 = model.predict(img_cat)
print(prediction1)
if prediction1 > 0.5:
    print("Dog")
else:
    print("Cat")

img_cat1 = load_img('./data/cnn_dog_cat/cat1.jpeg', target_size=(50, 50))
img_cat1 = img_to_array(img_cat1)
img_cat1 = img_cat1/255
img_cat1 = img_cat1.reshape(1, 50, 50, 3)
prediction2 = model.predict(img_cat1)
print(prediction2)
if prediction2 > 0.5:
    print("Dog")
else:
    print("Cat")

#print the class indices
print(training_set.class_indices)