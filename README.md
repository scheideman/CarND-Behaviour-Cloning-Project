# CarND-Behaviour-Cloning-Project
End-to-End learning for self driving car using simulated images

The submission includes a model.py file, drive.py, model.h5 and a writeup report report.pdf.
## Instuctions:
- Get simulator:https://github.com/udacity/self-driving-car-sim
- Run simulator in autonomous mode
- Predict steering angles using model: `python drive.py model.h5`

## Preprocessing:

####Generator:  
- A generator was used for feeding data to the network so that the entire image dataset did not 
have to be held in memory. With a python generator, the function will wait at any yeild statements
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
- For image normalization a image Keras lambda layer was used to normalize the image pixels between -0.5-0.5
- model.py line 51 and line 123
``` python 
def normalize_image(image):
    image = image / 255 - 0.5
    return image
...
model.add(Lambda(normalize_image,input_shape=(64,64,3)))
```
- Normalizing the images in the DNN means you don't have to worry about normalization images when testing the model
later

#### Cropping
- The top 50 pixels are cropped to remove the horizon from the image, and the bottom 25 pixels are cropped to remove 
the car hood from the image. Since the horizon and hood don't help the car drive they were removed.
- model.py, line 58:
``` python
image = image[50:(image.shape[0]-25), 0:image.shape[1]]
```

#### Data augmentation




