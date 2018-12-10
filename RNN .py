#@Author Schinaid

import warnings
warnings.filterwarnings("ignore")
#import cientifico de operação
import numpy as np
#tratamento de Imagem
from PIL import Image
#import para plotagem dos graficos e etc
import matplotlib 
#import para DeepLearnin Rnn e o krl a 4
import tensorflow 
from keras.models import Sequential, model_from_json, Model
from keras.layers import MaxPooling2D, Embedding, BatchNormalization
from keras.layers import Convolution2D, Dense, Dropout, InputLayer, Flatten, LSTM, Input, concatenate
from keras.layers.merge import concatenate
#modelos
import h5py
#importação para a merda do diretorio
import os
from os import listdir
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
filters = 12
batch_size = 32
train = 'data/train/'
teste = 'data/teste/'

train_datagen = ImageDataGenerator(rotation_range=40,#rotação na imagem
                                   rescale=1./255,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,#zoom
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(directory=train,
                                                    target_size=[100, 100],
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    follow_links=True)

validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(directory=teste,
                                                              target_size=[100, 100],
                                                              batch_size=batch_size,
                                                              class_mode='binary',
                                                              follow_links=True, 
                                                              shuffle=False)

x_train,y_train=train_generator.next()
x_teste, y_teste=validation_generator.next()
np.shape(x_train)


#convolução
classifier = Sequential()

#Entrada
#classifier.add(InputLayer(input_shape=(100, 100, 3)))
classifier.add(Convolution2D(32,3,3, input_shape = (100,100,3), activation ='tanh'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2), input_shape = (100,100,3)))
classifier.add(Dropout(0.2))
#layers 1
classifier.add(Convolution2D(64,3,3, activation = 'tanh'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2), input_shape = (100,100,3)))
classifier.add(Dropout(0.1))
#layers 2
classifier.add(Convolution2D(128,3,3,  activation = 'tanh'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2), input_shape = (100,100,3)))
classifier.add(Dropout(0.2))
#layers 3
classifier.add(Convolution2D(256,3,3,  activation = 'tanh'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2), input_shape = (100,100,3)))
classifier.add(Dropout(0.2))
classifier.add(Flatten())

#saida
classifier.add(Dense(units = 8196, activation = 'tanh'))
classifier.add(Dropout(rate = 0.2))
classifier.add(Dense(1, activation = 'sigmoid'))

image_input = Input(shape=(100, 100, 3))
encoded_image = classifier(image_input)

question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=100, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)
merged = concatenate([encoded_question, encoded_image])


classifier.compile(loss=['binary_crossentropy'], optimizer='Adam', metrics=['accuracy', 'mae', 'categorical_accuracy', 'binary_accuracy'])
classifier.summary()

history=classifier.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=50)

model_json = classifier.to_json()
with open("model/modelo_.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model/modelo_.h5')  
classifier.save('model/modelo_.h5')
classes = train_generator.class_indices
print(classes)
print("Modelo Salvo com sucesso!!")

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Treinamento')
plt.ylabel('precisão')
plt.xlabel('Epoca')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
  
plt.plot(history.history['val_mean_absolute_error'])
plt.plot(history.history['mean_absolute_error'])
 
plt.title('Treinamento usando a métrica de erro absoluto')
plt.ylabel('Mean_absolute error')
plt.xlabel('Epoca')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Treinamento usando a métrica binaria')
plt.ylabel('binary_accuracy')
plt.xlabel('Epoca')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])    
plt.title('Treinamento usando a métrica de categoria')
plt.ylabel('categorical_accuracy')
plt.xlabel('Epoca')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

