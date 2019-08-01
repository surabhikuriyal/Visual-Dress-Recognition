import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import cv2

train = pd.read_csv('clothes.csv')    # reading the csv file
print(train.shape)
train_image = []
for i in tqdm(range(train.shape[0])):
    #print(train['photo'][i])
    img = image.load_img('clothes/'+train['photo'][i],target_size=(28,28,1),grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

y=train['ID'].values
y = to_categorical(y)
#Converts a class vector (integers) to binary class matrix #keras
print(y)
#print(type(y))

#sklearn model selection , Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
#checks validation on 10% of images

#model architechture
model = Sequential()
#filter, kernel size, 
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

score=model.evaluate(X_test, y_test)

print("loss and accuracy of test data",score)

import pickle
with open('my_model','wb')as f:
    pickle.dump(model,f)


