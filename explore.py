import cv2
import csv
from random import shuffle
import numpy as np
import random
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import math
from keras import backend as K
import copy
tf.python.control_flow_ops = tf
###
# Script for exploring dataset
###

def darken_image(img):
    image = copy.copy(img)    
    bright_factor = .25
    #assuming HSV image
    image[:,:,2] = image[:,:,2]*bright_factor
    
    return image

def flip_image(img, steering_angle):
    image = copy.copy(img)   
    image = cv2.flip(image,1)
    steering_angle = steering_angle * -1

    return image, steering_angle

def random_shadow(img):
    image = copy.copy(img)   
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
    cv2.imwrite("original.jpg",image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    #crop image 
    imageC = image[50:(image.shape[0]-25), 0:image.shape[1]]
    image1 = darken_image(image)    
    image2 = random_shadow(image)
    image3, y = flip_image(image,y)
    imageR = cv2.resize(image,(64,64),interpolation = cv2.INTER_AREA)
    
    imageC = cv2.cvtColor(imageC, cv2.COLOR_HSV2RGB)
    cv2.imwrite("cropped.jpg",imageC)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    cv2.imwrite("dark.jpg",image1)
    image2 = cv2.cvtColor(image2, cv2.COLOR_HSV2RGB)
    cv2.imwrite("shadow.jpg",image2)
    image3 = cv2.cvtColor(image3, cv2.COLOR_HSV2RGB)
    cv2.imwrite("flipped.jpg",image3)
    imageR = cv2.cvtColor(imageR, cv2.COLOR_HSV2RGB)
    cv2.imwrite("resized.jpg",imageR)
    
    return image,y

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

        
        x_center, y_center = preprocess_pipeline(x_center,y_center)
        cv2.imshow("left", x_left)
        cv2.imshow("center", x_center)
        cv2.waitKey(0)
        
