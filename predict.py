#@Author: Schinaid
# 

from keras.models import Sequential
from keras.layers import Flatten
from keras import optimizers
from keras import applications
from keras.preprocessing import image
from keras.optimizers import SGD
import keras
from tkinter import *
import warnings# leia abaixo
warnings.filterwarnings("ignore")
from keras.layers import Dense
from numpy import array
import numpy as np
from PIL import ImageTk, Image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator



img_width, img_height = 100, 100
classifier = load_model('model/modelo_.h5')
classifier_score = classifier
classifier = Sequential()
classifier.add(Flatten())
img = image.load_img('data/save/pc.jpg', target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = classifier.predict_classes(images)
score = np.amax(classifier_score.predict_proba(images,batch_size=1, verbose=1))
score = np.float64(score)

print(classes, score)

if classes >= [10000]:
    classes = ('Pessoa caida: {:.2f}% de precisão'.format(score))
elif classes <= [9000]:
    classes = ('pessoa em pé: {:.2f}% de precisão'.format(score))
else:    
    classes = 'problema não catalogado'

print(classes)