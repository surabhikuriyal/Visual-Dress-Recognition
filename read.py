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
#print(train.head())# printing first five rows of the file
#print(train.columns)

train_image = []
for i in tqdm(range(train.shape[0])):
    #print(train['photo'][i])
    img = image.load_img('clothes/'+train['photo'][i],target_size=(28,28,1),grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

#print(X)
#print(X.shape)
#print(train_image)

#plt.imshow(X[2])

#print(train['breed'][2])

y=train['ID'].values
y = to_categorical(y)
print(type(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

#model.summary()

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

score=model.evaluate(X_test, y_test)
#print('Test loss:', score['loss'], "\n")
#print('Test accuracy:', score['acc'], "\n")
print("loss and accuracy of test data",score)

#test=pd.read_csv('test.csv')
'''test_image=[]
for i in tqdm(range(test.shape[0])):
    print(test['photo'][i])
    img = image.load_img('input/'+test['photo'][i],target_size=(28,28,1),grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)
# making predictions
prediction=model.predict_classes(test, batch_size=32, verbose=1)
#prediction = model.predict_classes(test)
#test['label']=prediction
#test.to_csv('test.csv',header=True,index=False)'''

import os

path='input'
imagePaths=[os.path.join(path,f)for f in os.listdir(path)]
for p1 in imagePaths:
    test_image = []
    a=cv2.imread(p1)
    cv2.imshow('test image',a)
    img = image.load_img(p1,target_size=(28,28,1),grayscale=True)
    img = image.img_to_array(img)
    #print(img)
    img1 = img/255
    #test_image.append(img)
    #test = np.array(test_image)
    #print(test)
    classes=np.array(train.columns[1])
    prediction = model.predict_classes(img.reshape(1,28,28,1))
    #print(prediction[0])
    #print(np.sort(prediction[0])[:])
    #top=np.argsort(prediction[0])[:]
    #print(top)
    #print(len(top))
    #print(int(prediction))
    #print(type(prediction))
    #print(len(prediction))
    p=int(prediction)
    if p==1:
        print(p1,"   shirt",end="")
        print('')
    if p==2:
        print(p1,"   trouser",end="")
        print('')
    if p==3:
        print(p1,"   boots",end="")
        print('')
    if p==4:
        print(p1,"   gloves",end="")
        print('')
#for i in range(0,len(top)):
#    print("{}".format(prediction[0][top[i]]))



#print(accuracy_score(y, prediction,normalize=False))
