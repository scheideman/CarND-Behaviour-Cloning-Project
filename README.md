# CarND-Behaviour-Cloning-Project
End-to-End learning for self driving car using simulated images

The submission includes a model.py file, drive.py, model.h5 and a writeup report in this README.
## Instuctions:
- Get simulator:https://github.com/udacity/self-driving-car-sim
- Run simulator in autonomous mode
- Predict steering angles using model: `python drive.py model.h5`

## Preprocessing:

- An example starting image:   
![Alt text](https://github.com/scheideman/CarND-Behaviour-Cloning-Project/blob/master/examples/original.jpg?raw=true "Cropped Image")

####Generator:  
- A generator was used for feeding data to the network so that the entire image dataset did not 
have to be held in memory. With a python generator, the function will wait at any yield statements
until the generator function is called again. This way only one batch of images is saved in memory.
- model.py, line 82-116: 
``` python 
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
               #Load images and preprocess
               ...
            
            X_train, y_train = sklearn.utils.shuffle(np.array(X_train), np.array(y_train))
            yield ({'lambda_input_1': X_train}, {'dense_4': y_train})
```

####Image normalization:
- For image normalization a Keras lambda layer was used to normalize the image pixels between -0.5 and 0.5
- model.py line 51 and line 123
``` python 
def normalize_image(image):
    image = image / 255 - 0.5
    return image
...
model.add(Lambda(normalize_image,input_shape=(64,64,3)))
```
- Normalizing the images with a keras layer means you don't have to worry about normalization images when testing the model
later

#### Cropping:
- The top 50 pixels are cropped to remove the horizon from the image, and the bottom 25 pixels are cropped to remove 
the car hood from the image. Since the horizon and hood don't help the car drive they were removed. Cropping made the car not wondering as much.
- model.py, line 58:
``` python
image = image[50:(image.shape[0]-25), 0:image.shape[1]]
```
![Alt text](https://github.com/scheideman/CarND-Behaviour-Cloning-Project/blob/master/examples/cropped.jpg?raw=true "Cropped Image")

- The images were also resized to 64X64X3. This number was chosen since after cropping the height was already 65, so I also reduced the width to increase training time. This seemed to have no negative affect on the learned model.   

![Alt text](https://github.com/scheideman/CarND-Behaviour-Cloning-Project/blob/master/examples/resized.jpg?raw=true)

#### Left and right images
- Using the left and right camera images produced by the simulator was a easy way to increase the amount of training data, as well as simulate how to correct from bad positions. I experimented with different values for steering angle offset and found +- 25 to be good. So for left images you want to simulate correcting by steering right and with right images you correct by steering left. Line 102, model.py:
``` python 
 y_left = float(row[3]) + 0.25
 y_right = float(row[3]) - 0.25
 ```

#### Data augmentation:
- I used three methods for data augmentation: flipping the image, darkening some of the image, and adding 'shadows'. A great resource for this project is : https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.pjtjwn7p8 and is where I got the inspiration to add random shadows and changing image brightness. Adding shadows and darkening the image made the model more robust to shadows on the road.
- For flipping the images I used the opencv `cv2.flip` function and changed the sign of the steering angle. The function is on line 26 in model.py   
![Alt text](https://github.com/scheideman/CarND-Behaviour-Cloning-Project/blob/master/examples/flipped.jpg?raw=true)

- In order to augment the data to handle shaded portions of the road I scaled the brightness of the HSV V channel image. After experimenting I found 0.25 scaling to work well. Line  22 of model.py:
``` python 
bright_factor = .25
image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
image[:,:,2] = image[:,:,2]*bright_factor
```
![Alt text](https://github.com/scheideman/CarND-Behaviour-Cloning-Project/blob/master/examples/dark.jpg?raw=true)

- For adding shadows I randomly created regions about half the size of the image and then applied the same process as above for darkening the region. After experimenting I found always scaling the image brightness by 0.3 worked good. Line 35, model.py;
``` python 
bright_factor = 0.3
x = random.randint(0, image.shape[1])
y = random.randint(0, image.shape[0])

width = random.randint(int(image.shape[1]/2),image.shape[1])
if(x+ width > image.shape[1]):
    x = image.shape[1] - x
height = random.randint(int(image.shape[0]/2),image.shape[0])
if(y + height > image.shape[0]):
    y = image.shape[0] - y
image[y:y+height,x:x+width,2] = image[y:y+height,x:x+width,2]*bright_factor
```
![Alt text](https://github.com/scheideman/CarND-Behaviour-Cloning-Project/blob/master/examples/shadow.jpg?raw=true)

 - The entire preprocessing pipeline on line 55 of model.py:
 ``` python 
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
```

## Model
- I initially started by recreating the model in this paper: https://arxiv.org/abs/1604.07316. And was able to successfully get the car to drive around the lake track but not the mountain track. After that I wanted to do some experimentation with different architectiures with less weights and modified the network to get:    
`INPUT -> [CONV (stride 2) -> ELU -> Dropout]*3 -> [FC -> ELU -> Dropout]*3 -> OUTPUT`
- This model has less layers and generalized better to the unseen track. For the convolutional layers I used 3X3 filters and started with 24 depth then increased to 48 and 96 in the following two layers. All three of the convolutional layers downsample the image using a stride of two. For the fully connected layers I flatten the final convolutional layer to a 4704 length vector and then use 500, 50, 10, 1 for the next layers, with the last being the predicted steering angle. 
- For a activation function I started with RELU, but decided to try Keras implementation of the Exponential Linear Unit (ELU). Unlike RELU the ELU does not output zero for negative inputs so will not suffer the vanishing gradient problem. After experimentation I found ELU made the model generalize better as it performed much better on the mountain track.  
- For input into the model 64X64 images were used.

#### Regularization
- I used dropout on the convolutional layers and then L2 weight adjustment for both the convolutional layers and fully connected layers. Weights in the network were initialized using a normal distribution.

#### Training & Testing 
- For the loss function I used mean squared error and used the ADAM learning algorithm. The Adam optimizer is nice because it automatically adjusts the learning rate over epochs unlike SGD.
- I used ~150,000 images (left, right, center) and split them 90/10 for the training/validation sets. See line 79 in model.py
- The images used I created in the simulator(lake track) using a combination of driving in the center of the road as well as recovering from bad positions. In addition I drove both directions around the track, and I also used the left and right camera images. 
- For testing I used the mountain track.

#### Other Hyperparameters
- 7 epochs over ~45000 images were used. I found this number from trial and error. Anymore epochs seemed to cause overfitting. 
- For the L2 regularization I used 0.0001 for all layers except the first fully connected layer, which I used 0.0003 for instead as it had a lot more weights and was more prone to overfitting. I found these values from trial and error. 

## Conclusion
- The learned model drives safely on both lake track and the mountain track.
- Lake track: https://youtu.be/eecYTdzYAfY
- Mountain track: https://youtu.be/HtZX0ISo890
- I would typically run the car at 20 mph when testing the model in autonomous mode, any faster and the car would sometimes get stuck in a over correction loop and start swerving from one side of the road to the other. 
- This was a difficult project which I spent a lot of time fine tuning. But I learned a lot and it has introduced me to using  Convolutional Neural Networks for problems besides classification.



