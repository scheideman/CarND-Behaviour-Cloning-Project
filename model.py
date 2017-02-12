import numpy as np
import csv
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
import tensorflow as tf
import random
import math
import sklearn
tf.python.control_flow_ops = tf

def darken_image(image):    
    bright_factor = .25
    #assuming HSV image
    image[:,:,2] = image[:,:,2]*bright_factor
    
    return image

def flip_image(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = steering_angle * -1

    return image, steering_angle

def random_shadow(image):
    bright_factor = 0.3
    
    x = random.randint(0, image.shape[1])
    y = random.randint(0, image.shape[0])

    width = random.randint(int(image.shape[1]/2),image.shape[1])
    if(x+ width > image.shape[1]):
        x = image.shape[1] - x
    height = random.randint(int(image.shape[0]/2),image.shape[0])
    if(y + height > image.shape[0]):
        y = image.shape[0] - y
    
    #Assuming HSV image
    image[y:y+height,x:x+width,2] = image[y:y+height,x:x+width,2]*bright_factor

    return image


def normalize_image(image):
    image = image / 255 - 0.5
    return image

def preprocess_pipeline(image, y):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #crop image 
    image = image[50:(image.shape[0]-25), 0:image.shape[1]]
    
    if(random.random() <= 0.4):
        image = darken_image(image)    
    if(random.random() <= 0.4):
        image = random_shadow(image)
    
    if(random.random() <= 0.8):
        image, y = flip_image(image,y)
    
    image = cv2.resize(image,(64,64),interpolation = cv2.INTER_AREA)
    return image,y


samples = []
path = '../recording/driving_log.csv'

with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    samples = list(reader)

train_samples, validation_samples = train_test_split(samples, test_size=0.1)


def data_generator(samples, batch_size = 50):
    while 1:
        random.shuffle(samples)
        print(len(samples))
        num_samples = len(samples)
        X_train = []
        y_train = []
        count = 0
        for offset in range(0, num_samples, batch_size):
            #for row in file_reader:
            
            batch_samples = samples[offset:offset+batch_size]
            X_train = []
            y_train = []
            for row in batch_samples:
                x_center = cv2.imread(row[0])
                x_left = cv2.imread(row[1].strip())
                x_right = cv2.imread(row[2].strip())
                y_center = float(row[3])
                y_left = float(row[3]) + 0.25
                y_right = float(row[3]) - 0.25

                x_center,y_center = preprocess_pipeline(x_center,y_center)
                x_left,y_left = preprocess_pipeline(x_left,y_left)
                x_right,y_right = preprocess_pipeline(x_right,y_right)
                
                X_train.append(x_center)
                y_train.append(y_center)
                X_train.append(x_left)
                y_train.append(y_left)
                X_train.append(x_right)
                y_train.append(y_right)
            
            X_train, y_train = sklearn.utils.shuffle(np.array(X_train), np.array(y_train))
            yield ({'lambda_input_1': X_train}, {'dense_4': y_train})


# Create the Sequential model
model = Sequential()

model.add(Lambda(normalize_image,input_shape=(64,64,3)))

# 3X3 convolution layer
model.add(Convolution2D(24,3,3,
                        border_mode='valid',
                        input_shape=(64,64,3),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 3X3 convolution layer
model.add(Convolution2D(48,3,3,
                        border_mode='valid',
                        input_shape=(31,31,24),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 3X3 convolution layer
model.add(Convolution2D(96,3,3,
                        border_mode='valid',
                        input_shape=(15,15,96),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

model.add(Flatten(input_shape=(7, 7, 96)))

model.add(Dense(500,
                W_regularizer=l2(0.0003),
                init='normal'))
model.add(ELU(alpha=1.0))

model.add(Dense(50,
                W_regularizer=l2(0.0001),
                init='normal'))
model.add(ELU(alpha=1.0))

model.add(Dense(10,
                W_regularizer=l2(0.0001),
                init='normal'))
model.add(ELU(alpha=1.0))

model.add(Dense(1,
                W_regularizer=l2(0.001),
                init='normal'))

model.compile(optimizer=Adam(lr=0.0001), loss = 'mse', metrics=['mean_absolute_error'])
print("Done compiling")

history = model.fit_generator(data_generator(train_samples),
                            validation_data=data_generator(validation_samples),
                            nb_val_samples=len(validation_samples),
                            samples_per_epoch=len(train_samples), nb_epoch=7)


model.save('model.h5')



