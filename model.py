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

#Retrieved from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ga5cuizax
def random_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    
    random_bright = .25
    
    image[:,:,2] = image[:,:,2]*random_bright
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def flip_image(image, steering_angle):
    image = cv2.flip(image,1)
    steering_angle = steering_angle * -1

    return image, steering_angle


def random_shadow(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    random_bright = 0.45
    
    x = random.randint(0, image.shape[1])
    y = random.randint(0, image.shape[0])

    width = random.randint(int(image.shape[1]/2),image.shape[1])
    if(x+ width > image.shape[1]):
        x = image.shape[1] - x
    height = random.randint(int(image.shape[0]/2),image.shape[0])
    if(y + height > image.shape[0]):
        y = image.shape[0] - y
    
    image[y:y+height,x:x+width,2] = image[y:y+height,x:x+width,2]*0.15
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)

    return image

def normalize_image(image):
    image = image / 255 - 0.5
    return image

def preprocess_pipeline(image, y):
    #crop image (1/4 if the top and 25 pixels from the bottom)
    image = image[math.floor(image.shape[0]/5):image.shape[0]-25, 0:image.shape[1]]
    
    if(random.random() <= 0.4):
        image = random_brightness(image)    
    if(random.random() <= 0.4):
        image = random_shadow(image)
    
    if(random.random() <= 0.6):
        image, y = flip_image(image,y)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image = cv2.resize(image,(160,80),interpolation = cv2.INTER_AREA)
    return image,y

#'recording/driving_log.csv'
def generate_arrays_from_csv(path, batch_size = 40):
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
                #if(isOn and float(row[3]) == 0):
                    #file_reader.remove(row)
                #    isOn = False
                #    continue
                #elif(float(row[3]) == 0):
                #    isOn = True

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
                    #yield ({'convolution2d_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)})
                    #yield ({'batchnormalization_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)})
                    yield ({'lambda_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)})
                    #yield ({'dropout_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)})
                    count = 0
                    X_train = []
                    y_train = []
                else:
                    count += 1
            #yield ({'batchnormalization_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)}) 
            #yield ({'convolution2d_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)})
            yield ({'lambda_input_1': np.array(X_train)}, {'dense_4': np.array(y_train)})
            
        #yield ({'convolution2d_input_1': np.array([x_center,x_left,x_right])}, {'dense_4': np.array([y_center,y_left,y_right])})
        #yield ({'input_1': x_center, 'input_2': x_left,'input_3': x_right}, {'output': np.array([y_center,y_left,y_right])})



# Create the Sequential model
model = Sequential()

model.add(Lambda(normalize_image,input_shape=(80,160,3)))
#model.add(BatchNormalization(input_shape=(80,160,3),mode=2))
#model.add(Dropout(0.1, input_shape=(80,160,3)))
#model.add(Dropout(0.05))

# 5X5 convolution layer
model.add(Convolution2D(24,5,5,
                        border_mode='valid',
                        input_shape=(80,160,3),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 5X5 convolution layer
model.add(Convolution2D(36,5,5,
                        border_mode='valid',
                        input_shape=(38,78,24),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 5X5 convolution layer
model.add(Convolution2D(48,5,5,
                        border_mode='valid',
                        input_shape=(17,37,36),
                        subsample=(2,2),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 3X3 convolution layer
model.add(Convolution2D(64,3,3,
                        border_mode='valid',
                        input_shape=(7,17,48),
                        subsample=(1,1),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

# 3X3 convolution layer
model.add(Convolution2D(64,3,3,
                        border_mode='valid',
                        input_shape=(5,15,64),
                        subsample=(1,1),
                        W_regularizer=l2(0.0001),
                        init='normal'))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

model.add(Flatten(input_shape=(3, 13, 64)))

model.add(Dense(200,
                W_regularizer=l2(0.0002),
                init='normal'))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))

model.add(Dense(50,
                W_regularizer=l2(0.0001),
                init='normal'))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))

model.add(Dense(10,
                W_regularizer=l2(0.0001),
                init='normal'))
model.add(Activation('relu'))
#model.add(ELU(alpha=1.0))

model.add(Dense(1,
                W_regularizer=l2(0.0001),
                init='normal'))

model.compile(optimizer=Adam(lr=0.0001), loss = 'mse', metrics=['mean_absolute_error'])
print("Done compiling")
#history = model.fit(X_train, y_train, batch_size=100, nb_epoch=5, validation_split=0.2)

history = model.fit_generator(generate_arrays_from_csv('../recording/driving_log.csv'),
                            validation_data=generate_arrays_from_csv('../recording/driving_log_val.csv'),
                            nb_val_samples=4000,
                            samples_per_epoch=40000, nb_epoch=7)


model.save_weights('model.h5')
json_string = model.to_json()
with open("model.json", "w") as text_file:
    text_file.write(json_string)



