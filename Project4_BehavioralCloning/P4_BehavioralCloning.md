## Udacity Self Driving Car Project 4 - Behavioral Cloning
Author: Tomas Mawyin
 
#### The objective of this project is to use Neural Networks and Udacity's autonomous driving simular to perform behavioral cloning to teach the simulator vehicle to drive by itself.

---

**Goals and Steps of the project:**

* Dataset Collection & Exploration: Approach on collecting data, understand images, and data distrubition from the simulator
* Data Augmentation: Generating more images to represent a larger training dataset 
* Pre-processing & Model Architecture: Image pre-processing techniques to facilite the model architecture and development of a neural network architecture using Keras as a regression model
* Model Training: Optimize parameters to train the model and obtain accurate results
* Model Testing: Use the model to successfully drive the vehicle in Autonomous mode

[//]: # (Image References)

[image1]: ./writeup_images/simulator.png "Udacity Simulator"

[image2]: ./writeup_images/left.jpg "Left Camera"
[image3]: ./writeup_images/center.jpg "Center Camera"
[image4]: ./writeup_images/right.jpg "Right Camera"

[image5]: ./writeup_images/hist1.jpeg "Steering Angle Distribution"

[image6]: ./writeup_images/hist2.jpeg "Data Augmentation"

[image7]: ./writeup_images/nvidiann.png "Nvidia Architecture"

[image8]: ./writeup_images/loss.jpeg "Loss vs Epoch"

---

#### The purpose of this document is to describe the steps that have been implemented for the creation of the behavioral cloning project. I will provide examples and code snippets in each of the steps to show how the code works.

### Data Collection & Exploration

As with any project involving neural networks and images, the first step is to understand the datasets in preparation to generate a good pre-processing strategy and a good model architecture. To capture the images and labels I make use of the Udacity simulator in "Training" mode (see image below). This allow us to drive the vehicle around the track and capture images from the left, center, and right cameras as well as the steering angle.  

![alt text][image1]

I want to make sure I captured enough data for our model to learn. Typically we want the model to learn by having enough data to generalize enough so that the car can drive in any scenario. In order to do this, I collected data 95% of the data from the required track and 5% of the data from the second track. In addition, I collected data from difficult corners in the first track to adjust the model response during tough turns. Finally, I collected some data driving in the opposite direction on the track (clock-wise) to avoid introducing any biases.

After driving in training mode, the images will become our features while the steering value will become the label. The simulator also prints out a log file in CSV format that will be used as the input file in the code. We started by loading the datasets using the `csv` library. Note in the following code snippet how the library is used to open the file, parse through it (we skip the header) and save all the data into a list variable called `lines`.

```Python
import csv 
def load_csv():
    # Opening the CVS file and gathering all lines
    lines = []
    with open(log_path) as csvfile:
        reader = csv.reader(csvfile)
        # Let's skip the header: ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        next(reader)
        for line in reader:
            # lines contains all the cvs file in a list
            lines.append(line)

    return lines
```

The function above returns a list containing **18572** rows of data. Each row contains the filename of the left, center, and right images as well as the steering angle. This means that we have **55716** images in total to work with. Now that we know how many features and labels we have, let's take a look at the features images. The images below are an example of the left, center, and right images:

| Location   | Image |
|------------|-------|
| Left       | ![alt text][image2] |
| Center     | ![alt text][image3] |
| Right      | ![alt text][image4] |

Images are *320x160* pixels in size with 3 channels (R, B, and G). Also, note how the images depict not only the road but there are other features like the trees, sky, and scenary that might confuse our model, so a good strategy for pre-processing will be to crop the images to only capture road features.

Now, we can take a look at the labels. The figure below is a representation of the distribution of labels taken from the first drive on the track:

![alt text][image5]

There are a few things to point out: First, the range of the steering angle is normalized from [-1, 1]. Second, the distribution is very biased towards zero degrees. This means that most of the time is spent driving at zero degree steering angle. Unfortunately, this might be a problem if we want the model to generalize the driving behaviour to more agressive tracks with more turns and slopes. One way to work around this issue is to generate additional images and augment the training dataset.

### Data Augmentation

Data augmentation is the process of generating new images to increase the training dataset. In our case, the best way to generate new images is to flip the image to get a different perspective. In order to do this, we make use of OpenCV's `flip()` function. We flip the images along the horizontal but in order to maintain the structure we also change the sign of the steering angle. The following code performs this action:

```Python
# For steering angles other than zero, we randomize flipping an image
if (float(batch_sample[3]) != 0) and (random.randint(0,1) == 1):
    # images are flipped and appended to the image_input array
    image_input.append(cv2.flip(image,1))
    # Steering angle is negated and appended to the steering_label array
    steering_label.append(-1*float(batch_sample[3]))
```

In the code, I made use of the left, center, and right images. When using the left and right camera images we need to adjust for the steering angle since the position of the camera is off from where we would like to be. We use a correction factor of 0.25 to correct for the offset of the camera location. Note that the images are flipped in a random order and also only images that have a non-zero steering angle get flipped during training. After performing the above code on the initial training set, we get the following distribution:

![alt text][image6]

Note that the distribution grew around 0 and +|-0.25 where the correction factor is. When the code uses all the image sets the distribution should be more evely distributed. 

### Pre-proscessing & Model Architecture

I have already discussed some of the strategies to pre-process the images. Thanks to Keras, it is possible to combine some of the pre-processing part with the model architecture. In terms of the model architecture, I followed the suggested Nvidia architecture as outlined in the image below:

![alt text][image7]

To capture the architecture, I generated a function that opens a Sequential() model and returns the final model. As stated before, the initial steps in the model is to perform some pre-processing:

```Python
# Doing some preprocessing: cropping image 60 top, 25 bottom 
model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
# Doing some preprocessing: resizing the image to match nvidia model
model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66,200)))) # resize image
# Doing some preprocessing: normalizing data
model.add(Lambda(lambda x: (x / 255) - .5))
```

First we crop the image. This is done with the `Cropping2D` function. In this function we cut 60 pixels from the top of the image to get rid of all the unnecessary scenary in the image and keep only the road. Similarly, we crop the last 25 pixes from the bottom of the image since we can see the car hood and we don't want to confuse the model.

The second step is to resize the images. This accomplishes two things, helps us follow the original Nvidia model and helps us speed up the training of the model since we use a smaller image. 

The final step in pre-processing is to normalize all the images. Note that I make use of the `Lambda()` function from Keras to apply the resizing and normaliztion, this makes it easier to perform such actions to all the images during training.

A summary of the model with the output shape at each stage and the number of parameters used is given here:

```Python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 75, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
lambda_2 (Lambda)            (None, 66, 200, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
dropout_1 (Dropout)          (None, 31, 98, 24)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
dropout_2 (Dropout)          (None, 14, 47, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
dropout_3 (Dropout)          (None, 5, 22, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
dropout_4 (Dropout)          (None, 3, 20, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
dropout_5 (Dropout)          (None, 1, 18, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1164)              1342092   
_________________________________________________________________
dense_2 (Dense)              (None, 100)               116500    
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_4 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 11        
=================================================================
Total params: 1,595,511
Trainable params: 1,595,511
Non-trainable params: 0
```

Finally, note that the model contains some `Dropout` layers as I wanted to make sure the model was not overfitting. I used a dropout ratio of 0.1 which means only 10% of the probabilities are being dropped after each Convolution layer. Another thing to point out is that the model uses the "ELU()" activation function to account for non-linearities. Unlike the RELU activation function, ELU can produce negative outputs and sometimes is a stronger alternative.

With the model generated, now we are ready to train the model and obtain our steering predictions.

### Model Training

Before training the model, we need to consider a few things: 1) there needs to be a good way of measuring "accuracy" of the model which is typically done by separating the data into training and validation sets. 2) Since we have a large dataset, it is important to make sure we apply a method to load images in batches, for this we make use of a "generator". 3) Finally, a good set of parameters is required in particular the number of EPOCHS and the BATCH number.

The first point of separating the data is easily achieved with the `sklearn` library. In particular we use the following code to break the information from the CSV in two parts; 80% of the data becomes the training set while the other 20% becomes the validation set: 

```Python
from sklearn.model_selection import train_test_split

# Splitting data into training and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
```

Now that we have a good data split, we can focus on one of the most important parts of the model efficiency. You can imaging how loading more than 50K images could be very memory intensive for any processor. For this reason we make use of a generator - this will allow us to load images in batches and train the model as we go. The following code is a smaller version of the generator used in the code:

```Python
def generator(samples, batch_size=100, train=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            image_input = []
            steering_label = []
            
            for batch_sample in batch_samples:
                # CENTER IMAGES
                filename = batch_sample[0].split('/')[-1]
                image = cv2.imread(img_path+filename)[...,::-1]
                image_input.append(image)
                steering_label.append(float(batch_sample[3]))

                if train == True:
                    # Flipping some center images
                    if (float(batch_sample[3]) != 0) and (random.randint(0,1) == 1):
                        image_input.append(cv2.flip(image,1))
                        steering_label.append(-1*float(batch_sample[3]))

            # trim image to only see section with road
            X_train = np.array(image_input)
            y_train = np.array(steering_label)
            yield shuffle(X_train, y_train)
```

The generator function gets the training or validation sets. For the training set, we set the `train` variable as "TRUE". This will help us flip the images and also it will load the left and right camera images (not shown in the code above). For the validation set, we don't need to flip any images since we only need to validate the model on the images from the center camera. As we move throught the each item on the `batch_samples`, we load the images and the steering angle and we append them to their respective lists. Finally, we convert the lists into arrays, we shuffle them and then we "YIELD" them to the model. Once the model has completed training on this batch, the generator function yields the next batch and so on.

Now, we can set up some of the parameters, run the model, save it and test it in the simulator in autonomous mode. The following snippet of the code shows how the model is compiled and run:

```Python
# compile and train the model using the generator function
batch_size = 100
train_generator = generator(train_samples, batch_size=batch_size, train=True)
validation_generator = generator(validation_samples, batch_size=batch_size, train=False)

model = nn_model()

# Compiling and Running the model with or without generators
model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, batch_size=batch_size)
history_object = model.fit_generator(train_generator, steps_per_epoch= np.ceil(len(train_samples)/batch_size),validation_data=validation_generator, 
    validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=5, verbose = 1)
```

A few things to note. First, I set up the `batch_size` to 100. This takes means we take 100 images at a time at each EPOCH. I selected the number based on some trial and error, 100 seems to be a good fit for the model although it is possible to increase the number a bit showing promising results. The next thing to note is how we pass the generator using the train and validation samples and using the `fit_generator()` function. Note that I use only 5 EPOCHS as this seems to be enough to run the model with good results.

It can be seen that the model compiles using the Adam stochastic optimizer where I maintain the default learning rate of 0.001. Also, since this is a regression model we use the mean squared error (MSE) as the loss function.

### Model Testing

We test the model by allowing it to run its course and keep track of the Loss vs Epoch. The following graph and results display shows how the training and validation loss decay with every EPOCH. 

```Python
Epoch 1/5
149/149 [==============================] - 94s - loss: 0.0635 - val_loss: 0.0287
Epoch 2/5
149/149 [==============================] - 80s - loss: 0.0416 - val_loss: 0.0273
Epoch 3/5
149/149 [==============================] - 80s - loss: 0.0385 - val_loss: 0.0272
Epoch 4/5
149/149 [==============================] - 80s - loss: 0.0367 - val_loss: 0.0268
Epoch 5/5
149/149 [==============================] - 80s - loss: 0.0353 - val_loss: 0.0280
```

![alt text][image8]

Finally, here is a [link to the video result](video.mp4)

---

### Discussion

Implementing this code was very fun and it was very exciting to learn about this amazing application of machine learning and neural networks. This is a great example of how machine learning can help us represent the real world. It is important to note that Keras makes the architecture generation very easy to implement and even how easy it becomes to generate some of the pre-processing functions. 

One of the improvements I would do to the model would be to augment the data even more. I think, I can implement other augmentation techniques such as rotating, translating, blurring, or changing the brightness to the images. This will help the model to generalize a bit more in other conditions such as dark roads. I would also like to generate more data using other tracks in order to have a more generic dataset. Even though I used information from the second track, only 5% of the images were from this track so the model does not generalize that well in this second track.

Another thing I would try would be to implement other model architectures. Maybe we can reduce the number of layers to speed up the training process or even try other techniques such as L2 regularization. Finally, I would like to play a bit more with some of the parameters on the model, chaning the EPOCHS or the Dropout rate are good ways to measure accuracy and convergence of the model.