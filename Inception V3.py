#import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
#import matplotlib.pyplot as plt
from glob import glob

#Resize all images
IMAGE_SIZE=[224,224]
train_path='train'
valid_path='valid'

#Import the Inception V3 library as shown below and add preprocessing layer to the front of Inception
#Here we will be using Imagenet Weights
inception=InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

#Don't treain exisiting weights
for layer in inception.layers:
    layer.trainable=False

#Geting the number of outputs class
folders=glob('train/*')
# Our layers we can add more if we want
x=Flatten()(inception.output)
prediction=Dense(len(folders),activation='softmax')(x)

#Create Model object
model=Model(inputs=inception.input, outputs=prediction)
model.summary()
#Cost adn Optimization method to use
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#use the image Data Generattor to import then images from dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)
traing_set=train_datagen.flow_from_directory('train',
                                             target_size=(224,224),
                                             batch_size=16,
                                             class_mode='categorical')
test_set=test_datagen.flow_from_directory('valid',
                                          target_size=(224,224),
                                          batch_size=16,
                                          class_mode='categorical')
r=model.fit_generator(traing_set,
                      validation_data=test_set,
                      epochs=15,
                      steps_per_epoch=len(traing_set),
                      validation_steps=len(test_set))
model.save('InceptionV3.h5')




