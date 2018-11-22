# PROJECT 3 - TRAFFIC SIGN CLASSIFIER
# Creating a Traffic Sign Classificer using a similar method as LeNet
# This is a training Python file. The real version is in the iPython Notebook
# Github: https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project

import pickle
import numpy as np 

# ----- LOADING DATA
# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file= './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ----- LOADING THE CSV FILE
import pandas as pd 
signnames = pd.read_csv('signnames.csv')


# ----- FEATURES AND LABELS
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# ----- VISUALIZATION
import matplotlib.pyplot as plt
import random

# Selecting a random image
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

# plt.figure(figsize=(1,1))
# plt.imshow(image)
# plt.tight_layout()
# plt.show()

# Printing the image label and the signame name
print('The label is: ', y_train[index])
print(signnames.loc[signnames["ClassId"].isin([y_train[index]]), "SignName"].tolist()[0])

# Visualizing multiple images - grid of 5x5
# fig, ax = plt.subplots(5,5, sharex=True, sharey=True)
# for row in range(5):
# 	for col in range(5):
# 		index = random.randint(0, len(X_train))
# 		image = X_train[index].squeeze()
# 		ax[row,col].imshow(image)
# fig.subplots_adjust(hspace=0, wspace=0)
# plt.show()

# Finding the class distributions
# plt.hist(y_train, bins=n_classes, alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Classes')
# plt.ylabel('Frequency')
# plt.title('Distribution of Classes on Train data')
# plt.tight_layout()
# plt.show()


# ----- DEBUGGING: SELECT RANDOM DATA FOR TESTING MODEL
# Backup of initial data
X_train_backup = X_train
y_train_backup = y_train
X_valid_backup = X_valid
y_valid_backup = y_valid

# Selection 1000 points for training set
rnd_train = np.random.choice(X_train.shape[0], 1000, replace = False)
X_train = X_train[rnd_train,:]
y_train = y_train[rnd_train]

# Selection 250 points for training set
rnd_valid = np.random.choice(X_valid.shape[0], 250, replace = False)
X_valid = X_valid[rnd_valid,:]
y_valid = y_valid[rnd_valid]

# Finding the class distributions
# plt.hist(y_train, bins=len(np.unique(y_train)), alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Classes')
# plt.ylabel('Frequency')
# plt.title('Distribution of Classes on Sample Train data')
# plt.show()

print("Length of training set =", X_train.shape[0])
print("Total number of classes =", len(np.unique(y_train)))
# -----


# ----- PRE-PROCESSING DATA
import cv2

# GRAY SCALE: This converts images to grayscale in an "average method" 
def gray_image(image_set):
	dst = np.sum(image_set/3, axis=3, keepdims=True)
	return dst

# NORMALIZE IMAGE
def norm_image(image):
	dst = (image - 128)/128
	return dst

# POSITION TRANSLATION
def translate_image(image):
	h, w = image.shape[:2]
	px = 2
	dx, dy = np.random.randint(-px,px,2)

	trans_M = np.float32([[1,0,dx],[0,1,dy]])
	dst = cv2.warpAffine(image, trans_M, (h,w))

	# print(dst.shape)
	# plt.figure(figsize=(1,1))
	# plt.imshow(dst)
	# plt.tight_layout()
	# plt.show()

	return dst

# SCALE IMAGE [0.9 to 1.1]
def scaling_image(image):
	n = 0.2*np.random.uniform()+0.9 # Caviat, if n =0.5 then image is the same
	res = cv2.resize(image,None,fx=n, fy=n, interpolation = cv2.INTER_CUBIC)
	new_h, new_w = res.shape[:2]
	pad = (image.shape[0] - res.shape[0])
	p = int(np.abs(pad)/2)
	# We will check if we need equal padding or cropping
	if pad % 2 == 0:
		# Check if we need cropping (<0) or padding (>0)
		if pad < 0:
			# print("Need even crop ", res.shape)
			dst = res[0+p:new_h-p, 0+p:new_w-p]
		else:
			# print("Need even pad ", res.shape)
			dst = cv2.copyMakeBorder(res, p, p, p, p, borderType=cv2.BORDER_CONSTANT, value=0)
	else:
		if pad < 0:
			# print("Need odd crop ", res.shape)
			dst = res[0+p:new_h-(p+1), 0+p:new_w-(p+1)]
		else:
			# print("Need odd pad ", res.shape)
			dst = cv2.copyMakeBorder(res, p, p+1, p, p+1, borderType=cv2.BORDER_CONSTANT, value=0)

	# print(dst.shape)
	# print(np.array_equal(image,dst))

	# plt.figure(figsize=(1,1))
	# plt.imshow(dst)
	# plt.tight_layout()
	# plt.show()

	return dst

# ROTATE IMAGE
def rotate_image(image):
	h, w = image.shape[:2]
	angle = np.random.uniform(-15, 15)

	rot_M = cv2.getRotationMatrix2D((h/2,w/2), angle, 1)
	dst = cv2.warpAffine(image, rot_M, (h,w))

	# print(dst.shape)
	# plt.figure(figsize=(1,1))
	# plt.imshow(dst)
	# plt.tight_layout()
	# plt.show()

	return dst

# AFFINE TRANSFORMATION
def affine_image(image):
	h, w = image.shape[:2]

	shear_rate = np.random.randint(-3,3)
	pt1 = 8 + shear_rate*np.random.uniform() - shear_rate/ 2
	pt2 = 15 + shear_rate*np.random.uniform() - shear_rate/ 2
	pt3 = 20 + shear_rate*np.random.uniform() - shear_rate/ 2
	pts1 = np.array([[8, 15], [20, 8], [20, 20]]).astype('float32')
	pts2 = np.float32([[pt1, pt2], [pt3, pt1], [pt3, pt3]])

	M = cv2.getAffineTransform(pts1,pts2)
	dst = cv2.warpAffine(image, M, (h,w))

	# print(np.array_equal(image,dst))
	# print(dst.shape)
	# plt.figure(figsize=(1,1))
	# plt.imshow(dst)
	# plt.tight_layout()
	# plt.show()

	return dst

# BRIGHTNESS & CONTRAST CHANGE
def brightness_image(image):
	alpha = np.random.uniform(1.01,3) # Simple contrast control (1 for normal)
	beta = np.random.uniform(0.1,50)  # Simple brightness control (0 for normal)

	dst = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

	# print(np.array_equal(image,dst))
	# plt.figure(figsize=(1,1))
	# plt.imshow(dst)
	# plt.tight_layout()
	# plt.show()

	return dst

# BLUR IMAGE
def blur_image(image):
	blur_rate = np.random.randint(1, 3)
	kernel = 2*blur_rate + 1
	dst = cv2.GaussianBlur(image, (kernel, kernel), 0)

	# print(np.array_equal(image,dst))
	# print(dst.shape)
	# plt.figure(figsize=(1,1))
	# plt.imshow(dst)
	# plt.tight_layout()
	# plt.show()

	return dst

# CREATES NEW IMAGE - Randomizes image effect
def new_image(image,choice=[1,3]):
	dst = image
	for i in choice:
		if i == 1:
			dst = translate_image(dst)
		elif i == 2:
			dst = scaling_image(dst)
		elif i == 3:
			dst = rotate_image(dst)
		elif i == 4:
			dst = affine_image(dst)
		elif i == 5:
			dst = brightness_image(dst)
		elif i == 6:
			dst = blur_image(dst)
		else:
			pass
	
	if (dst.shape[0] != 32) or (dst.shape[1] != 32) or (dst.shape[2] != 3):
		dst = image

	return dst


# Generating fake data to improve accuracy of model
n_classes = len(np.unique(y_train))
n_sample = 100

for i in range(n_classes):
	# Separating the training set based on classes 
	X_new = X_train[y_train==i]
	X_size = len(X_new)
	if X_size < n_sample:
		for n in range(n_sample-X_size):
			# Selecting a radom image to perform transformation
			rnd = np.random.randint(0,X_size)
			# Select two random transformations
			choice = np.random.choice([1,2,3,4,5,6],2,replace=False)
			# Evaluate both transformations
			dst = new_image(X_new[rnd], choice)
			dst = np.expand_dims(dst, axis = 0)
			# Append these new images to the train set
			X_train = np.append(X_train,dst, axis = 0)
			# Append the label of this new image to the label set
			y_train = np.append(y_train, i)

print("After adding images, new training size = ", X_train.shape[0])

# Convert data to grayscale
X_train_gray = gray_image(X_train)
X_valid_gray = gray_image(X_valid)
X_test_gray = gray_image(X_test)

# Here we will do some preprocessing to ensure the data is more normalized
# Normalizing pixels in the data based on the [0,255] values
X_train_norm = norm_image(X_train_gray)
X_valid_norm = norm_image(X_valid_gray)
X_test_norm = norm_image(X_test_gray)

print("Train dataset new shape =", X_train_norm.shape)
print("Train dataset new mean =", np.mean(X_train_norm))
print("Train dataset new standar deviation =", np.std(X_train_norm))

# Let's shuffle the data a bit more to avoid any biases
from sklearn.utils import shuffle
X_train_norm, y_train = shuffle(X_train_norm, y_train)

# ----- MODEL ARCHITECTURE
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Defining hyperparameters
EPOCHS = 5
BATCH_SIZE = 100
rate = 0.001
dropout_conv = 1.0 # Droupout rate parameter
dropout_layer = 0.95 # Droupout rate parameter
beta = 0.01	# In case of L2-regularization
n_input = X_train_norm.shape[3]

def TrafficSign_LeNet(x, drop_conv=1.00, drop_fc=0.95):
	# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    # Note that the filter size is (5x5), together with a "VALID" padding, and the stride size, we can get the output size
    # Recall the formula: out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, n_input, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation function is the RELU() function.
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.dropout(conv1, drop_conv)

    # Max Pooling. Input = 28x28x6. Output = 14x14x6.
    # Note that stride and ksize of 2 is a way to reduce the output from 28 to 14
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.dropout(conv2, drop_conv)

    # Max Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten into a vector. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, drop_fc)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, drop_fc)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

# ----- FEATURES AND LABELS
# x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
x = tf.placeholder(tf.float32, (None, 32, 32, n_input))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

drop_conv = tf.placeholder(tf.float32)
drop_fc = tf.placeholder(tf.float32)

# ----- TRAINING PIPELINE
# Here are the basic steps of the neural network.
logits = TrafficSign_LeNet(x, drop_conv, drop_fc)

# Cross Entropy measure difference between the logits (calculated) with the labels (given)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)

# Loss operation is calculated by averaging the cross-entropy for all training sets in the batch
loss_operation = tf.reduce_mean(cross_entropy)
# Adding L2-Regularization, we adapt the loss_operation
# l2_reg = tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc2_W) + tf.nn.l2_loss(fc3_W)
# loss_operation = tf.reduce_mean(loss_operation + l2_reg * beta)

# Minimizing the loss function
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
# Backpropagation is used here where we minimize the optimizer and update weights
training_operation = optimizer.minimize(loss_operation)

# ----- EVALUATION PIPELINE
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# Evaluate function averages the accuracy over all batches to find the final accuracy
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, drop_conv: 1.0, drop_fc: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# ----- TRAIN THE MODEL
# - Run the training data through the training pipeline to train the model.
# - Before each epoch, shuffle the training set.
# - After each epoch, measure the loss and accuracy of the validation set.
# - Save the model after training. 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_norm)
    validation_numbers = []
    
    print("Training...")
    print()
    for i in range(EPOCHS):
    	# Shuffle data again to avoid order biases
        X_train_norm, y_train = shuffle(X_train_norm, y_train)
        # Break the model into batches and apply the Train Pipeline on each batch
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_norm[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, drop_conv: dropout_conv, drop_fc: dropout_layer})
        
        # Passing the validation set and findining the accuracy after each EPOCH
        validation_accuracy = evaluate(X_valid_norm, y_valid)
        validation_numbers.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './SaveData/lenet')
    print("Model saved")

# Plotting Validation Accuravy vs EPOCHS
plt.plot(validation_numbers)
plt.ylabel('Validation Accuracy')
plt.show()

# ----- EVALUATE THE MODEL
# Evaluation the performance of the model in the Test data set
# Only to be done once the training and validation achieve a high accuracy

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./SaveData/.'))

    test_accuracy = evaluate(X_test_norm, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    print("")

# -----
# ----- WORKING WITH NEW IMAGES
import glob
import matplotlib.image as mpimg

y_newLabels = [9,4,14,17,12,34,13,25,23,40]

X_newImg = np.empty([0,32,32,3])

# Opening the files and setting up an array of images
files = sorted(glob.glob('new-images/*.jpeg'))
for f in files:
	img = mpimg.imread(f)
	img = cv2.resize(img,(32,32), interpolation = cv2.INTER_CUBIC)
	img = np.expand_dims(img, axis = 0)
	X_newImg = np.append(X_newImg, img, axis = 0).astype(np.uint8)

# Pre-processing the image (gray-scale and normalizing pixels)
X_newImg = gray_image(X_newImg)
X_newImg = norm_image(X_newImg)

# Visualizing multiple images - grid of 2x5
fig, ax = plt.subplots(2,5, sharex=True, sharey=True)
for row in range(2):
	for col in range(5):
		image = X_newImg[row*5+col].squeeze()
		ax[row,col].imshow(image, cmap="gray")
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()

# Prediction on new images
new_prediction = tf.argmax( logits, 1 )
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./SaveData/.'))

    output_prediction = sess.run(new_prediction, feed_dict={x: X_newImg, drop_conv: 1.0, drop_fc: 1.0})
    print("New Images Prediction = ", output_prediction)
    print("")

# Testing accuracy on new images
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./SaveData/.'))

    test_accuracy = evaluate(X_newImg, y_newLabels)
    print("New Images Test Accuracy = {:.3f}".format(test_accuracy))

# Finding the top_k probabilities
top_k = tf.nn.top_k(tf.nn.softmax(logits), k=5)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./SaveData/.'))
    output_top_k = sess.run(top_k, feed_dict={x: X_newImg, drop_conv: 1.0, drop_fc: 1.0})

# Printing top 5 probabilities
for i in range(10):
	print("Image ", i+1)
	print("Probabilities = ", output_top_k.values[i])
	print("Corresponding labels ", output_top_k.indices[i])
	print("")

# Visualizing the probabilities for each new image
fig, ax = plt.subplots(10,2, figsize=(10,20))
for row in range(10):
	for col in range(2):
		image = X_newImg[row].squeeze()
		ax[row,0].imshow(image, cmap="gray")
		ax[row,1].bar(output_top_k.indices[row],output_top_k.values[row], align='center', alpha=0.5 )
		ax[row,1].set_xlim([0,42])
		ax[row,1].set_ylim([0,1])
		ax[row,1].set_ylabel('Probability')
		ax[row,1].set_title("Correct Label = {}".format(y_newLabels[row]))
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()