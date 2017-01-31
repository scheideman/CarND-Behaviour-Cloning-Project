import cv2
import csv
from random import shuffle
import numpy as np



with open('recording/driving_log.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    spamreader = list(spamreader)
    shuffle(spamreader)
    for row in spamreader:
        ##print(row[0])
        ##print(row[1].strip())
        ##print(row[2])
        ##print(row[3])
        x_center = cv2.imread(row[0])
        x_left = cv2.imread(row[1].strip())
        x_right = cv2.imread(row[2].strip())
        y_center = float(row[3])
        y_left = float(row[3]) + 0.25
        y_right = float(row[3]) - 0.25

        x_center = cv2.resize(x_center,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        x_left = cv2.resize(x_left,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        x_right = cv2.resize(x_right,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        
        image1 = cv2.cvtColor(x_center,cv2.COLOR_RGB2HSV)
        random_bright = .25+np.random.uniform()
        image1[:,:,2] = image1[:,:,2]*random_bright
        x_center_b = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    
        x_center = cv2.cvtColor(x_center, cv2.COLOR_RGB2YUV)
        #x_center_b = x_center[:,:,2] * .25+np.random.uniform()

        np.reshape(x_center,(1,x_center.shape[0],x_center.shape[1],x_center.shape[2]))
        np.reshape(x_left,(1,x_left.shape[0],x_left.shape[1],x_left.shape[2]))
        np.reshape(x_right,(1,x_right.shape[0],x_right.shape[1],x_right.shape[2]))
        print(x_center.shape)
        ##print(x_center.shape)
        #print(type(x_center))
        stacked = np.array([x_center,x_left,x_right])

        print(stacked.shape)

        cv2.imshow("left", x_left)
        cv2.imshow("center", x_center)
        cv2.imshow("right", x_center_b)
        cv2.waitKey(0)
        break
