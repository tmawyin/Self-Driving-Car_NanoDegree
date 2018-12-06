# TERM 1 - PROJECT 4
# Testing Main Pipeline Code 

import csv
import os
import cv2
import random
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ----- LOADING CSV FILE(S)
# Setting up some path variables
img_path = '../../opt/carnd_p3/my_data/IMG/'
log_path = '../../opt/carnd_p3/my_data/driving_log.csv'

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
# -----

'''
# ----- NO GENERATORS
def img_load_ng(lines):
	# Gathering all images
	image_input = []
	steering_label = []

	for line in lines:
		# CENTER IMAGES:
		# Taking only the pathname (last element of the split)
		filename = line[0].split('/')[-1]
		# Opening the images
		image = cv2.imread(img_path+filename)#[...,::-1]
		image_input.append(image)
		steering_label.append(float(line[3]))

		if float(line[3]) != 0:
			image_input.append(cv2.flip(image,1))
			steering_label.append(-1*float(line[3]))

		# LEFT IMAGES (0.25 correction):
		filename = line[1].split('/')[-1]
		# Opening the image
		image = cv2.imread(img_path+filename)#[...,::-1]
		image_input.append(image)
		steering_label.append(float(line[3])+0.25)

		# RIGHT IMAGES (0.25 correction):
		filename = line[2].split('/')[-1]
		# Opening the image
		image = cv2.imread(img_path+filename)#[...,::-1]
		image_input.append(image)
		steering_label.append(float(line[3])-0.25)

	# Convert the image features and the steering angle class into numpy arrays
	X_train = np.array(image_input)
	y_train = np.array(steering_label)

	return X_train, y_train
# -----
'''

# ----- GENERATORS
# The following function defines the generator
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

					# LEFT IMAGES (0.25 correction):
					filename = batch_sample[1].split('/')[-1]
					# Opening the image
					image = cv2.imread(img_path+filename)[...,::-1]
					image_input.append(image)
					steering_label.append(float(batch_sample[3])+0.25)

					# Flipping some left images
					if (float(batch_sample[3]) != 0) and (random.randint(0,1) == 1):
						image_input.append(cv2.flip(image,1))
						steering_label.append(-1*(float(batch_sample[3])+0.25))

					# RIGHT IMAGES (0.25 correction):
					filename = batch_sample[2].split('/')[-1]
					# Opening the image
					image = cv2.imread(img_path+filename)[...,::-1]
					image_input.append(image)
					steering_label.append(float(batch_sample[3])-0.25)

					# Flipping some right images
					if (float(batch_sample[3]) != 0) and (random.randint(0,1) == 1):
						image_input.append(cv2.flip(image,1))
						steering_label.append(-1*(float(batch_sample[3])-0.25))

            # trim image to only see section with road
			X_train = np.array(image_input)
			y_train = np.array(steering_label)
			yield shuffle(X_train, y_train)
# -----

'''
# ------ VISUALIZATION
def visualize(X_train,y_train):
	# Getting a random image to plot
	random_img = np.random.randint(0,len(X_train))
	image = X_train[random_img]
	# Changing image from BGR TO RGB. Image shape is (160,320,3)
	image = image[...,::-1]
	plt.imshow(image)
	plt.show()

	# Getting a histogram of steering data
	plt.hist(y_train, 100, alpha=0.75)
	plt.grid(True)
	plt.show()
#-----
'''

# ----- BUILDING A NEURAL NETWORK
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Cropping2D, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.core import K

def nn_model():
	# Build the model
    dpout = 0.1
	model = Sequential()

	# Doing some preprocessing: cropping image 60 top, 25 bottom 
	model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=(160, 320, 3)))
	# Doing some preprocessing: resizing the image to match nvidia model
	model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66,200)))) # resize image
	# Doing some preprocessing: normalizing data
	model.add(Lambda(lambda x: (x / 255) - .5))

	# First convolution layer
	model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
	model.add(Dropout(dpout))

	# Second convolution layer
	model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
	model.add(Dropout(dpout))

	# Third convolution layer
	model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
	model.add(Dropout(dpout))

	# Fourth convolution layer
	model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(Dropout(dpout))

	# Fifth convolution layer
	model.add(Conv2D(64, (3, 3), activation='elu'))
	model.add(Dropout(dpout))

	# Fully connected layers
	model.add(Flatten())
	model.add(Dense(1164, activation='elu'))
	model.add(Dense(100, activation='elu'))
	model.add(Dense(50, activation='elu'))
	model.add(Dense(10, activation='elu'))
	model.add(Dense(1))

	return model
# -----


# ----- MAIN CODE
lines = load_csv()

# Splitting data into training and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
batch_size = 100
train_generator = generator(train_samples, batch_size=batch_size, train=True)
validation_generator = generator(validation_samples, batch_size=batch_size, train=False)

model = nn_model()
# model.summary()

# Compiling and Running the model with or without generators
model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, batch_size=batch_size)
history_object = model.fit_generator(train_generator, steps_per_epoch= np.ceil(len(train_samples)/batch_size),validation_data=validation_generator, 
	validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=5, verbose = 1)

# Saving the model
model.save('model_nn.h5')
print("")
print('model saved!')

# Plotting Epoch vs Loss
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.jpeg')
