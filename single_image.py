from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle

with open('my_model','rb')as f:
    model=pickle.load(f)

train = pd.read_csv('clothes.csv')

#FOR SINGLE IMAGE
test_image = []
p1='input/49-shirt.jpg'
#img = image.load_img(p1,target_size=(28,28,1),grayscale=True)
img = image.load_img(p1,target_size=(28,28,1),color_mode = "grayscale")
img = image.img_to_array(img)
classes=np.array(train.columns[1])
prediction = model.predict_classes(img.reshape(1,28,28,1))
p=int(prediction)
if p==1:
    v='SHIRT'
if p==2:
    v='TROUSER'
if p==3:
    v='BOOTS'
if p==4:
    v='GLOVES'

a=cv2.imread(p1)
cv2.putText(a, v, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3,8)

# display the output image
plt.imshow(a)
plt.show()
