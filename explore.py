import cv2
import csv
from random import shuffle
import numpy as np
import random
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import math
from keras import backend as K
tf.python.control_flow_ops = tf
###
# Script for exploring dataset
###
with open('recording/driving_log.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    spamreader = list(spamreader)
    shuffle(spamreader)
    for row in spamreader:

        x_center = cv2.imread(row[0])
        x_left = cv2.imread(row[1].strip())
        x_right = cv2.imread(row[2].strip())
        y_center = float(row[3])
        y_left = float(row[3]) + 0.25
        y_right = float(row[3]) - 0.25

        image1 = cv2.cvtColor(x_center,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        image1[:,:,2] = image1[:,:,2]*random_bright
        x_center_b = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    
        x_center = cv2.cvtColor(x_center, cv2.COLOR_RGB2YUV)

        np.reshape(x_center,(1,x_center.shape[0],x_center.shape[1],x_center.shape[2]))
        np.reshape(x_left,(1,x_left.shape[0],x_left.shape[1],x_left.shape[2]))
        np.reshape(x_right,(1,x_right.shape[0],x_right.shape[1],x_right.shape[2]))
        print(x_center.shape)
        stacked = np.array([x_center,x_left,x_right])

        shape = x_left.shape
        x_left = x_left[math.floor(shape[0]/4):shape[0]-20, 0:shape[1]]
        print(x_left.shape)
        x_c = x_center / 255 - 0.5

        print(stacked.shape)

        x_left_flip = cv2.flip(x_left,1)
        cv2.imshow("left", x_left)
        cv2.imshow("center", x_left_flip)
        cv2.imshow("right", x_center_b)
        cv2.waitKey(0)
        
