## Udacity Self Driving Car Project 3 - Traffic Sign Classifier
Author: Tomas Mawyin
 
#### The objective of this project is to use Neural Networks to classify common traffic signs. This project involves the use of Convolution neural networks and the German traffic sign dataset. 

---

**Goals and Steps of the project:**

* Dataset Exploration: Understand images and how the data is distributed
* Data Preprocessing: Chose different pre-processing techniques to facilite the model architecture
* Model Architecture: Develop a basic neural network architecture as a logistic classifier
* Model Training: Optimize parameters to train the model and obtain accurate results
* Model Testing: Use new images to test the model and obtain good predictions 

[//]: # (Image References)

[image1]: ./writeup_images/training_set_example.png "Training Dataset"
[image2]: ./writeup_images/distribution.png "Distribution of Classes"

[image3]: ./writeup_images/lenet.png "LeNet CNN"

[image4]: ./writeup_images/accuracy.png "Model Accuracy"

[image5]: ./writeup_images/new_images.png "New Images"
[image6]: ./writeup_images/final_images.png "Final Accuracy"

---

#### The purpose of this document is to describe the steps that have been implemented for the creation of the traffic sign classifier. I will provide examples and code snippets in each of the steps to show how the code works.

### Data Exploration

#### Objective: Summarize the datasets and understand data design

The first step in generating any neural network is to understand the data being used. In our case, we use three different datasets: the largest one for training the model, a validation set to understand accuracy during training, and a final testing set that will be used once all parameters are optimized as a final accuracy descriptor of the model.

I started by loading the datasets using the `pickle` library. Note in the following code snippet how the library is used to open the file and set two main variables: `X_train` and `y_train` which correspond to the matrix containg all images for training and their labels respectively.

```Python
training_file = '../data/train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']
```

I performed the same for the validation and testing sets. Of course, the next step is to take a look at the sizes of these datasets to obtain the following:

| Set        | Size  |
|------------|-------|
| Training   | 34799 |
| Validation | 4410  |
| Testing    | 12630 |

The data also shows that each image is of size 32x32x3 (meaning 3 channels - R,G,B) and that there are 43 different types of signs to classify.

Finally, it is always a good idea to plot the data and see the different images that compose the dataset. The figure below shows a basic grid of images taken from the training dataset. As you can see, the images are all different and our neural network will have to deal with recognizing darker, lighter, angled, and different sizes of traffic signs.

![alt text][image1]

In these situations, it is a good idea to understand the distribution of the data. This means that it is necessary to see how much data we have for each of the classes we were given in the training set. Having a large disparity in the distribution can influence the way the classifier behaves by becoming more bias towards those traffic signs with the most amount of images. The distribution of these data is shown in the figure below:

![alt text][image2]

As it can seen from the data there is really no equal distribution of these data. This will influence our classifier and decrease our accuracy for those traffic signs that are not well represented here. In order to fix this issue I will augment some of the data in the next step.

### Data Preprocessing

Data preprocessing is a data mining technique to make sure the data is understadable to our classifier. The minimum we would need to do is ensure that the data is normalized and there is not a very large range of pixel values that can confuse our neural network. We will perform this but as stated before, we need to take care of the data imbalance we have in the training dataset.

To eliminate the imbalance in the data I made use of the following six image manipulation methods using OpenCV:

- (1) Image translation: this is a geometric transformation of the image that shifts pixels on an image. I translate the image randomly in a range of -2px to 2px.
- (2) Image scaling: another geometric transformation that manipulates the size of the image. The image size is change from 90% to 110% of it's original image. For images that become smaller, we add additional padding to ensure the size remains at 32x32. For those images that increase in size, there is a small cropping procedure.
- (3) Image rotation: this is another geometric transformation that rotates the image from its center. The rotation is done in an angle in the range of -15 to 15 degress.
- (4) Image shearing: this procedure allows the change the perspective at which the image is seen. It works by moving a triangle (3 points) on an image but maintaining the relationship between each of the sides. This procedure is somethis called an affine transformation. The image is sheared in a random procedure from -3 px to 3 px as a shear rate.
- (5) Image brightness & contrast: as the name implies, this procedure changes the image brightness and its contrast to provide a separate image.
- (6) Image blurring: this method adds an effect on the image by making the image slightly blury. I use the Gaussian blur method from OpenCV which takes a kernel number, in this case this kernel is randomized.


In order to use the above functions, a new additional function was created called `new_image()`. This function takes the image to be modified and two integer numbers (from 1 to 6) corresponding to two of the functions mentioned above. This means that a new image is always created by performing two modifications to the original image.

To generate the new images, a loop was created to run over each of the classes and top each class with enough images to reach a count of 500 images. Note that the number of images to generate is a parameter that can be changes but generating images is an expensive process. After the images are generated, they are added to the training set with their corresponding class to the label set.

The final steps of the data preprocessing part is to convert all images to grayscale and normalize the pixels. We made use of the two following functions to do so:

```Python
def gray_images(image_set):
    if image_set.shape[3] == 3:
        new_set = np.sum(image_set/3, axis=3, keepdims=True)
    else:
        print("Incorrect shape for image_set")
    return new_set

def norm_image(image_set):
    norm_set = (image_set - 128)/128
    return norm_set
```

Note that the gray scaling of the images is done with the average method, where we add all three R, G, and B channels and divide the value by 3. Each of the images goes from having a size of 32x32x3 to a size of 32x32x1. Finally, we use the `norm_image()` function to normalize the pixels values. Now we are ready to discuss the neural network architecture and its use.

### Model Architecture

The model follows the LeNet architecture as provided in the lectures. You can see an application of the different layers of this architecture in the figure below:

![alt text][image3]

Some basic modifications were done to this architecture to include a dropout after each of the hidden layers. I added a dropout value for each of the convolutions and one value for the fully connected network that followed. In general, the model follows the following layout:

- Step 1: Convolution Layer #1 that takes the feature map input of 32x32x1 to an output of 28x28x6. We use a filter of size 5x5, stride value of 1, and a "VALID" padding during the convolution
- Step 2: Apply a Rectified Linear Unit (RELU) activation function to maintain nonlinearities
- Step 3: Apply a dropout rate of 95%. Meaning probabilities of 0.95 or above are kept
- Step 4: Apply Max-Pooling to reduce the size of the features from 28x28x6 to 14x14x6. We use a stride and a k-size of 2
- Step 5: Convolution Layer #2 that takes the new input map from 14x14x6 to an output of 10x10x16. Similar to the first convolution layer, a filter of 5x5, stride of 1 and "VALID" padding were used
- Step 6: Apply another RELU() activation function
- Step 7: Apply a dropout rate of 95%. 
- Step 8: Apply Max-Pooling to reduce the size of the features from 10x10x16 to 5x5x16. We use a stride and a k-size of 2
- Step 9: Make use of TensorFlow's `flatten()` function to reshape the input to a flat vector of 400 features
- Step 10: Fully Connected Layer to tranform the feature map from 400 to 120
- Step 11: Apply another RELU() activation function
- Step 12: Apply a dropout rate of 95%.
- Step 13: Fully Connected Layer to tranform the feature map from 120 to 84
- Step 14: Apply another RELU() activation function
- Step 15: Apply a dropout rate of 95%.
- Step 16: Final Layer to convert feature map to the output class of 43 (logits)

From here on, some additional TensorFlow variables are setup but the model is ready to be put in use. In order to set up the training pipeline, we make use of the following Python code:

```Python
# ----- TRAINING PIPELINE
# Here are the basic steps of the neural network.
logits = TrafficSign_LeNet(x, drop_conv, drop_fc)

# Cross Entropy measure difference between the logits (calculated) with the labels (given)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

# Loss operation is calculated by averaging the cross-entropy for all training sets in the batch
loss_operation = tf.reduce_mean(cross_entropy)
# Adding L2-Regularization, we adapt the loss_operation
# l2_reg = tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc3_W)
# loss_operation = tf.reduce_mean(loss_operation + l2_reg * beta)

# Minimizing the loss function
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
# Backpropagation is used here where we minimize the optimizer and update weights
training_operation = optimizer.minimize(loss_operation)
```

The first function `TrafficSign_LeNet()` is the function that follows the 16 steps of the model architecture. After the logits are found, the cross entropy is setup, basically we are looking at how far off we are from the logit to our given one hot encoded labels. We set up a loss function as the mean of the cross entropy and we use Adam optimizer to minimuze this loss function. Finally, to evaluate the model we look at the accuracy of prediction with the labels provided in the training set by checking how many labels we actually match and how many we didnot match.

### Model Training

To train the model, we first separate the training set into batches of size 100. We make use of 50 Epochs, meaning that we loop over the training strategy 50 times for each of the batches in the training model, this will allow us to keep a high accuracy as we continue to run the learning method at higher number of epochs. Note that the model is run in a TensorFlow session by calling the `training_operation` variable with the batch input and labels as well as the dropout variables.

Now, at each Epoch loop we use the validation dataset to measure the accuracy of our training model. We use the `evaluate()` function to check our accuracy on this dataset. The following graph describes how the accuracy changes at each EPOCH:

![alt text][image4]

Note that the final validation accuracy of our model reaches **96%**. 

Now, we use the testing dataset to find the accuracy on the model using images that it has never seen before. To do this, we make use of the `evaluate()` function to compare our testing dataset labels to what the model predicts. 

The model produces a testing accuracy of **93%**

### Model Testing

In order to see how well the model works, I made use of 10 random images found online. Since the images are not the same size, I made use of OpenCV to work through each image and reshape it to the correct size. The following code shows how the images are opened, resized, and append to a testing array.

```Python
import glob
import matplotlib.image as mpimg

y_newLabels = [9,4,14,17,12,34,13,25,23,40]

X_newImg = np.empty([0,32,32,3])

# Opening the files and setting up an array of images
files = sorted(glob.glob('new_images/*.jpeg'))
for f in files:
    img = mpimg.imread(f)
    img = cv2.resize(img,(32,32), interpolation = cv2.INTER_CUBIC)
    img = np.expand_dims(img, axis = 0)
    X_newImg = np.append(X_newImg, img, axis = 0).astype(np.uint8)

# Pre-processing the image (gray-scale and normalizing pixels)
X_newImg = gray_images(X_newImg)
X_newImg = norm_image(X_newImg)
```

Note that I also create a vector containing the labels of each of the images. Similarly, I perform the preprocessing steps of converting all these images to grayscale and normalizing their pixels.  

![alt text][image5]

The image above shows the traffic signs that would be used as a testing method to the model. Visually, these images do not show any deformation, maybe some streching, blurriness, and some shadows. The images do show a lot of pixelation because of the resizing but this should not affect the accuracy of the model.

We now can predict the labels for these images and also evaluate the accuracy of the model. We do this by the following code:

```Python
# Prediction on new images
new_prediction = tf.argmax( logits, 1 )
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    output_prediction = sess.run(new_prediction, feed_dict={x: X_newImg, drop_conv: 1.0, drop_fc: 1.0})
    print("New Images Prediction = ", output_prediction)
    print("")

# Testing accuracy on new images
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_newImg, y_newLabels)
    print("New Images Test Accuracy = {:.3f}".format(test_accuracy))
    print("")
```

The result from the above code showed: **New Images Prediction =  [41  4 14 17 12 33 13 25 23 40]** and the accuracy reached **80%**. As you can see the model didnot fully detect all images, it missed two images, the first image which should have been label 9 (instead of 41) and the sixth image which should have been label 34 (instead of 33). The following figure shows the display of the softmax of each of the images:

![alt text][image6]


-------------------------------------

### Discussion

Implementing this code was very fun and it was very exciting to learn about the machine learning and neural networks in particular. This is a great example of how machine learning can help us represent the real world. 

One of the improvements I would do to the model would be to generate a larger amount of "fake" images. Since this process is very time consuming I was not able to generate as many as I wanted to but by doing so the model's accuracy would have improved significantly. 

Another improvement I would do to the model is to work on different architectures. Even though the model uses LeNet's architecture, some additional modifications could be implemented. This goes from adding additional convolution layers to using some of the output from the convolution layers as inputs to the fully connected layers. 

Similarly, I set up the model training portion to include L2 regularization but I prefered to use dropout instead. An additional thing to try would be to change the architecture and use the L2 regularization methodology to improve the accuracy on the model.