from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle

with open('my_model','rb')as f:
    model=pickle.load(f)

train = pd.read_csv('clothes.csv')

        
#FOR A LIST OF IMAGES :
import os
path='input'
imagePaths=[os.path.join(path,f)for f in os.listdir(path)]
l=len(imagePaths)
print(l)
print("\n\n\n ACTUAL CLOTH :    COLTH CLASSIFIED AS :\n")
c=0
for p1 in imagePaths:
    test_image = []
    img = image.load_img(p1,target_size=(28,28,1),color_mode = "grayscale")
    img = image.img_to_array(img)
    classes=np.array(train.columns[1])
    prediction = model.predict_classes(img.reshape(1,28,28,1))
    p=int(prediction)
    p1=p1.split('-')[1].split('.')[0]
    if p==1:
        if(p1=='shirt'):
            c=c+1
        print(p1,"                   shirt",end="")
        print('')
    if p==2:
        if p1=='trouser':
            c=c+1
        print(p1,"                   trouser",end="")
        print('')
    if p==3:
        if p1=='boots':
            c=c+1
        print(p1,"                   boots",end="")
        print('')
    if p==4:
        if p1=='glove':
            c=c+1
        print(p1,"                   glove",end="")
        print('')
acc=(c*100)/l
print("\n\n ACCURACY : ",acc)


