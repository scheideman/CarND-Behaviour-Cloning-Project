import numpy as np
import csv
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from random import shuffle
import tensorflow as tf
tf.python.control_flow_ops = tf

#'recording/driving_log.csv'
def generate_arrays_from_csv(path, batch_size = 128):
    while 1:
        with open(path, newline='') as csvfile:
            file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            file_reader = list(file_reader)
            shuffle(file_reader)
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

                x_center = cv2.resize(x_center,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                x_left = cv2.resize(x_left,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                x_right = cv2.resize(x_right,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

                X_train.append(x_center)
                y_train.append(y_center)
                X_train.append(x_left)
                y_train.append(y_left)
                X_train.append(x_right)
                y_train.append(y_right)
                if(count == (batch_size-1)):
                    yield ({'convolution2d_input_1': np.array([X_train])}, {'dense_4': np.array([y_train])})
                    count = 0
                    X_train = []
                    y_train = []
                else:
                    count += 1
            yield ({'convolution2d_input_1': np.array([X_train])}, {'dense_4': np.array([y_train])})
            
        #yield ({'convolution2d_input_1': np.array([x_center,x_left,x_right])}, {'dense_4': np.array([y_center,y_left,y_right])})
        #yield ({'input_1': x_center, 'input_2': x_left,'input_3': x_right}, {'output': np.array([y_center,y_left,y_right])})


X_train = []
y_train = []

with open('recording/driving_log.csv', newline='') as csvfile:
    file_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    print(type(file_reader))
    count=0
    file_reader = list(file_reader)
    print(type(file_reader))
    print(len(file_reader))
    shuffle(file_reader)
    for row in file_reader:
        x_center = cv2.imread(row[0])
        x_left = cv2.imread(row[1].strip())
        x_right = cv2.imread(row[2].strip())
        x_center = cv2.resize(x_center,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        x_left = cv2.resize(x_left,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        x_right = cv2.resize(x_right,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

     

        X_train.append(x_center)
        X_train.append(x_left)
        X_train.append(x_right)
        
        y_center = float(row[3])
        y_left = float(row[3]) + 0.25
        y_right = float(row[3]) - 0.25
        y_train.append(y_center)
        y_train.append(y_left)
        y_train.append(y_right)

        count += 1
        if(count == 4500):
            break

X_train = np.array(X_train)
print(X_train.shape)
y_train = np.array(y_train)
print(y_train.shape)


# Create the Sequential model
model = Sequential()
#model.add(Dropout(0.2))

# 5X5 convolution layer
model.add(Convolution2D(24,5,5,
                        border_mode='valid',
                        input_shape=(80,160,3),
                        subsample=(2,2),
                        #W_regularizer=l2(0.001),
                        init='normal'))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# 5X5 convolution layer
model.add(Convolution2D(36,5,5,
                        border_mode='valid',
                        input_shape=(38,78,24),
                        subsample=(2,2),
                        #W_regularizer=l2(0.001),
                        init='normal'))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# 5X5 convolution layer
model.add(Convolution2D(48,5,5,
                        border_mode='valid',
                        input_shape=(17,37,36),
                        subsample=(2,2),
                        #W_regularizer=l2(0.001),
                        init='normal'))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# 3X3 convolution layer
model.add(Convolution2D(64,3,3,
                        border_mode='valid',
                        input_shape=(7,17,48),
                        subsample=(1,1),
                        #W_regularizer=l2(0.001),
                        init='normal'))
model.add(Activation('relu'))

# 3X3 convolution layer
model.add(Convolution2D(64,3,3,
                        border_mode='valid',
                        input_shape=(5,15,64),
                        subsample=(1,1),
                        #W_regularizer=l2(0.001),
                        init='normal'))
model.add(Activation('relu'))


model.add(Flatten(input_shape=(3, 13, 64)))

model.add(Dense(100,
                #W_regularizer=l2(0.001),
                activation='relu', init='normal'))
#model.add(Activation('relu'))

model.add(Dense(50,
                #W_regularizer=l2(0.001),
                activation='relu',init='normal'))
#model.add(Activation('relu'))

model.add(Dense(10,
                #W_regularizer=l2(0.001),
                activation='relu',init='normal'))
#model.add(Activation('relu'))

model.add(Dense(1,
                #W_regularizer=l2(0.001),
                init='normal'))

model.compile(optimizer=Adam(lr=0.0001), loss = 'mse', metrics=['mean_absolute_error'])
print("Done compiling")
#history = model.fit(X_train, y_train, batch_size=100, nb_epoch=5, validation_split=0.2)

history = model.fit_generator(generate_arrays_from_csv('recording/driving_log.csv'),
                            validation_data=generate_arrays_from_csv('recording/driving_log_val.csv'),
                            nb_val_samples=4000,
                            samples_per_epoch=24000, nb_epoch=10)


model.save_weights('model.h5')
json_string = model.to_json()
with open("model.json", "w") as text_file:
    text_file.write(json_string)



