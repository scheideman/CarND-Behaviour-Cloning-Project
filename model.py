import numpy as np
import csv
import cv2
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
    #crop image (1/5if the top and 25 pixels from the bottom)
    image = image[math.floor(image.shape[0]/5):image.shape[0]-25, 0:image.shape[1]]
    
    if(random.random() <= 0.4):
        image = darken_image(image)    
    if(random.random() <= 0.4):
        image = random_shadow(image)
    
    if(random.random() <= 0.8):
        image, y = flip_image(image,y)
    
    

    image = cv2.resize(image,(80,80),interpolation = cv2.INTER_AREA)
    return image,y

def generate_arrays_from_csv(path, batch_size = 50):
    while 1:
        isOn = True
        with open(path, newline='') as csvfile:
            file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            file_reader = list(file_reader)
            random.shuffle(file_reader)
            print(len(file_reader))
            X_train = []
            y_train = []
            count = 0
            for row in file_reader:
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

                if(count == (batch_size-1)):
                    yield ({'lambda_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)})
                    count = 0
                    X_train = []
                    y_train = []
                else:
                    count += 1
            if(count > 0):
                yield ({'lambda_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)})



# Create the Sequential model
model = Sequential()

model.add(Lambda(normalize_image,input_shape=(80,80,3)))

# 3X3 convolution layer
model.add(Convolution2D(24,3,3,
                        border_mode='valid',
                        input_shape=(80,80,3),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 3X3 convolution layer
model.add(Convolution2D(48,3,3,
                        border_mode='valid',
                        input_shape=(39,39,24),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 3X3 convolution layer
model.add(Convolution2D(96,3,3,
                        border_mode='valid',
                        input_shape=(19,19,48),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 3X3 convolution layer
model.add(Convolution2D(128,3,3,
                        border_mode='valid',
                        input_shape=(9,9,96,
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

model.add(Flatten(input_shape=(4, 4, 128)))

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

history = model.fit_generator(generate_arrays_from_csv('../recording/driving_log.csv'),
                            validation_data=generate_arrays_from_csv('../recording/driving_log_val.csv'),
                            nb_val_samples=4000,
                            samples_per_epoch=45000, nb_epoch=7)


model.save_weights('model.h5')
json_string = model.to_json()
with open("model.json", "w") as text_file:
    text_file.write(json_string)



