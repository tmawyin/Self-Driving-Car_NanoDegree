# PROJECT 2 - ADVANCED LANE DETECTION
# AUTHOR - TOMAS MAWYIN
# This file contains help functions required for the main pipeline

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

### ---------- CAMERA CALIBRATION ---------- ###
# This function is used to find corners in a list of images. We will use this to calibrate camera
# Parameters:
	# files is the list of images to be used for calibration
	# nx is the number of inside corners in x
	# ny us the number of inside corners in y
def find_corners(files, nx=9, ny=6, debug=False):
	# Let's set two empty arrays to hold the img_points and obj_points
	img_points = []
	obj_points = []	

	# objp is a 3D array with coordinates in (x,y,z) that hold the corner locations of the undistorded image. 
	# Here the array is of 48 rows and 3 columns:
	objp = np.zeros((nx*ny,3),np.float32)
	# We create a grid of points coordinates using mgrid(). We will get points like (0,0,0), (0,1,0), ... (7,5,0)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

	# Iterate through the files
	for file in files:
		# Reading each image and finding corners
		img = mpimg.imread(file)

		# Convert to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		# Find the chessboard corners
		# The function finds the internal corners on a chessboard image
		ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

		# If corners are found, append 
		if ret == True:
			img_points.append(corners)
            #This should be the same for all calibration images since undistorted image is unique
			obj_points.append(objp)
			if debug == True:
				cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
				plt.imshow(img)
				plt.show()
	return img_points, obj_points

# This function returns the undistorted image based on the object and image points (from find_corners())
# Parameters:
	# img is the image to undistort
	# objpoints are the original points from an undistorted chessboard
	# imgpoints are the corner points found on distorted chessboard images - to be mapped to the objpoints
def undistort_image(img, objpoints, imgpoints):
    # cv2.calibrateCamera() maps the imgpoints to objpoints returning the camera matrix (mtx) and distortion coefficients (dist)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    # cv2.undistort() returns an undistorted image given the matrix and coefficients
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undist


### ---------- COLOR SPACES AND GRADIENTS ---------- ###
# This function computes the Sobel operator (gradient) in x or y dimensions and returns a binary_image with threshold applied
# Parameters:
	# img is the image layered (either gray scale, HLS-channels or HSV-channels)
	# orient is the orientation required for the Sobel operator
	# sobel_kernel is the kernel parameter
	# thresh is a tuple with the min and max thresholds respectively
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Sanity check: if image is not layered, i.e. has RGB values) then convert to gray
    if len(img.shape) > 2:
    	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude is between threshold values
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # plt.imshow(binary_output, cmap='gray')
    # plt.show()

    return binary_output

# This function computes the Sobel magniture and returns a binary_image with threshold applied
# Parameters:
	# img is the image layered (either gray scale, HLS-channels or HSV-channels)
	# sobel_kernel is the kernel parameter
	# mag_thresh is a tuple with the min and max thresholds respectively
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Sanity check: if image is not layered, i.e. has RGB values) then convert to gray
    if len(img.shape) > 2:
    	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude 
    sobelxy = np.sqrt(np.square(sobelx)+np.square(sobely))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobelxy/np.max(sobelxy))
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return binary_output

# This function computes the Sobel direction and returns a binary_image with threshold applied
# Parameters:
	# img is the image layered (either gray scale, HLS-channels or HSV-channels)
	# sobel_kernel is the kernel parameter
	# dir_thresh is a tuple with the min and max thresholds respectively
def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):
    # Sanity check: if image is not layered, i.e. has RGB values) then convert to gray
    if len(img.shape) > 2:
    	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1

    return binary_output

# This function computes the HLS selection and returns a binary_image with threshold applied
# Parameters:
	# img is the image layered (H, L, or S channel)
	# channel is the required channel to threshold
	# channel_thresh is a tuple with the min and max thresholds respectively
def hls_select(img, channel='s', channel_thresh=(0, 255)):
	# For regular channeled image, we just perform the thresholding
	binary_output = np.zeros_like(img)
	binary_output[(img >= channel_thresh[0]) & (img <= channel_thresh[1])] = 1
	
	# Sanity check: if image is not layered, i.e. has RGB values) then convert to HLS color space
	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		# Check which channel was selected and applying threshold
		if channel == 'h':
			H = img[:,:,0]
			binary_output = np.zeros_like(H)
			binary_output[(H >= channel_thresh[0]) & (H <= channel_thresh[1])] = 1
		elif channel == 'l':
			L = img[:,:,1]
			binary_output = np.zeros_like(L)
			binary_output[(L >= channel_thresh[0]) & (L <= channel_thresh[1])] = 1
		else:
			S = img[:,:,2]
			binary_output = np.zeros_like(S)
			binary_output[(S >= channel_thresh[0]) & (S <= channel_thresh[1])] = 1
	
	# plt.imshow(binary_output, cmap = 'gray')
	# plt.show()

	return binary_output

# This function computes the HSV selection and returns a binary_image with threshold applied
# Parameters:
	# img is the image layered (H, S, or V channel)
	# channel is the required channel to threshold
	# channel_thresh is a tuple with the min and max thresholds respectively
def hsv_select(img, channel='v', channel_thresh=(0, 255)):
	# For regular channeled image, we just perform the thresholding
	binary_output = np.zeros_like(img)
	binary_output[(img >= channel_thresh[0]) & (img <= channel_thresh[1])] = 1
	
	# Sanity check: if image is not layered, i.e. has RGB values) then convert to HLS color space
	if len(img.shape) > 2:
		img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		# Check which channel was selected and applying threshold
		if channel == 'h':
			H = img[:,:,0]
			binary_output = np.zeros_like(H)
			binary_output[(H >= channel_thresh[0]) & (H <= channel_thresh[1])] = 1
		elif channel == 's':
			S = img[:,:,1]
			binary_output = np.zeros_like(S)
			binary_output[(S >= channel_thresh[0]) & (S <= channel_thresh[1])] = 1
		else:
			V = img[:,:,2]
			binary_output = np.zeros_like(V)
			binary_output[(V >= channel_thresh[0]) & (V <= channel_thresh[1])] = 1

	return binary_output

# This function computes the HSV selection and returns a binary_image with threshold applied
# Parameters:
	# img is the undistorted image to complete thresholding on
def threshold_find(img):
	# Parameter
	ksize = 15 # Choose a larger odd number to smooth gradient measurements
	
	# Sobel thresholds
	gradx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 110))
	grady_binary = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(60, 110))
	mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(80, 200))
	dir_binary = dir_threshold(img, sobel_kernel=ksize, dir_thresh=(0.7,1.5))

	# Measuring yellow lines: This is based on the saturation and hue channels
	h_binary = hls_select(img,'h', channel_thresh=(10, 40))
	s_binary = hls_select(img,'s', channel_thresh=(110, 255))
	yellow_line = np.zeros_like(s_binary)
	yellow_line[(h_binary == 1)  & (s_binary == 1)] = 1

	# Measuring white lines: This is based on the lightness channel
	l_binary = hls_select(img,'l', channel_thresh=(200, 255))
	white_line = np.zeros_like(s_binary)
	white_line[(l_binary == 1) ] = 1

	# Now we can combine all the different thresholds:
	combined = np.zeros_like(gradx_binary)
	combined[((gradx_binary == 1) & (grady_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | ((yellow_line == 1) | (white_line == 1))] = 1

	return combined


### ---------- PERSPECTIVE TRANSFORM ---------- ###
# This function computes perspective transform and retuns a warped image
# Parameters:
	# img is the image to be warped
	# src are the source points from the image
	# dst are the destination points
def perspective_transform(img, src, dst):
	# Image size
	img_size = (img.shape[1], img.shape[0])
	# Obtaining the transformation matrix M
	M = cv2.getPerspectiveTransform(src, dst)
	# Applying transformation matrix to the image points
	warped = cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)
	return warped 


### ---------- LANE LINES  ---------- ###
# This function finds lanes pixels in an image
# Parameters:
	# img is the warped binary image
def find_lane_pixels(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set a counter to check if margin size is enough
    counter = 0

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for index, window in enumerate(range(nwindows)):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        # Find the four below boundaries of the window 
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin 
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        # The nonzero()[0] is required since we only want nonzero pixels within the rectangle and we need it as an array (not tuple)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window (`right` or `leftx_current`) on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

# This function will fit a polynomial given a warped image and return the image with lanes and the fitted line
# Parameters:
	# img is the warped binary image
def find_polynomial(img, debug=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(img)

    # Calling the fit_poly function to find x,y points for left and right sides
    left_fitx, right_fitx, ploty = fit_poly(img.shape, leftx, lefty, rightx, righty)

    # Checking if lines were found
    if left_fitx.size == 0 and right_fitx.size == 0:
    	line_found = False
    else:
    	line_found = True 

    # Visualization: Colors in the left (red) and right (blue) lane regions
    if debug == True:
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.show()

    return out_img, left_fitx, right_fitx, ploty, line_found

# This function will fit a second order polynomial given the x and y points
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit() 
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    
    # Evaluating the polyfit on the left and right sides
    try:
        left_fitx = np.poly1d(left_fit)(ploty)
        right_fitx = np.poly1d(right_fit)(ploty)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    return left_fitx, right_fitx, ploty

# This function searches around a polynomial given the polyfit coefficients
def search_around_poly(binary_warped, left_fit, right_fit, debug=False):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Setting the area of search based on activated x-values within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (np.poly1d(left_fit)(nonzeroy) - margin)) & (nonzerox < (np.poly1d(left_fit)(nonzeroy) + margin)))
    right_lane_inds = ((nonzerox > (np.poly1d(right_fit)(nonzeroy) - margin)) & (nonzerox < (np.poly1d(right_fit)(nonzeroy) + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # Checking if lines were found
    if left_fitx.size == 0 and right_fitx.size == 0:
    	line_found = False
    else:
    	line_found = True 
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image - This is used for testing
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image - This is used for testing
    if debug == True:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.show()
    
    return result, left_fitx, right_fitx, ploty, line_found

# Function to draw window areas given:
# Parameters:
    # width is the width of the window area
    # height is the height of the window area
    # img_ref is the image as reference to calculate the image size
    # center is the center location of where to start drawing (centroid)
    # level is the layer along the image y-axis
def window_mask(width, height, img_ref, center,level):
    # Creates an output image - made of zeros
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

# Function used to as an alternative method to window sliding - This function will search for lines by using convolution
def find_window_centroids(image, window_width=50, window_height=80, margin=100, debug=False):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    # l_sum adds all values of 3/4 of the bottom of the image in y-axi, from left to 1/2 image in x - addition is done in y, thus axis=0 
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    # l_center finds the center (x-point) of the largest convolution value. 
    # Note that the '-window/2' is used since the convolute array is larger by ~ window/2, pushing the x point by that much to the right
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    # r_sum and r_center are the right side points (similar to the left side above)
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # Convolve the window into the vertical slice of the image
        # image_layer holds the sum of pixels along y-axis for each of the layers throught the image x-axis
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        # Moving the window along the layer throughout the x-axis
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        # At the end, window_centroids contains all left and right points for each of the levels
        window_centroids.append((l_center,r_center))

        # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        # Go through each level and draw the windows    
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,image,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,image,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage= np.dstack((image, image, image))*255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

        #  Getting the left and right points
        leftx = np.where(l_points == 255)[1]
        lefty = np.where(l_points == 255)[0]
        rightx = np.where(r_points == 255)[1]
        righty = np.where(r_points == 255)[0]

        # Fitting line and plotting
        left_fitx, right_fitx, ploty = fit_poly(image.shape, leftx, lefty, rightx, righty)

        # Setting line_found as True
        line_found = True
        
        # Adding some plot for testing
        if debug == True:
            plt.imshow(output)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.show()
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        line_found = False

    return output, left_fitx, right_fitx, ploty, line_found


### ---------- CURVATURE  ---------- ###
# This function calculates the curvature of the lane and the offcenter position in meters
# Parameters:
    # current_fit are the current left and right polyfit coefficients
    # ploty is an array contain all the y points 
def measure_curvature_center(current_fit, ploty):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension
    
    # Separating the left and right coefficients from current_fit
    left_fit_cr = current_fit[0]
    right_fit_cr = current_fit[1]
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix+left_fit_cr[1])**2)**(3/2))/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix+right_fit_cr[1])**2)**(3/2))/np.absolute(2*right_fit_cr[0])

    # Calculating the off-center position on the lane
    offcenter_pixel = (np.poly1d(left_fit_cr)(y_eval) + np.poly1d(right_fit_cr)(y_eval))/2 - 640
    offcenter = offcenter_pixel * xm_per_pix
    
    return (left_curverad, right_curverad) , offcenter


### ---------- UNWARPING IMAGES ---------- ###
# This function returns and un-warped image
def unwrap_image(img, left_fitx, right_fitx, ploty, src=None, dst=None):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    if (src.size == 0 & dst.size == 0):
        newwarp = color_warp
    else:
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = perspective_transform(color_warp, dst, src)

    return newwarp

### ---------- OTHER FUNCTIONS ---------- ###
# This function writes the curvature and off-center text on an the final image
def write_image(result, radius_of_curvature, line_base_pos):
    # Formating numbers and text
    float_formatter = lambda x: "%.2f" % x
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Writing Curvature
    cv2.putText(result,'Left Curvature:',(20,100), font, 1.2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,str(float_formatter(radius_of_curvature[0]))+' m',(330,100), font, 1.2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,'Right Curvature:',(20,150), font, 1.2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,str(float_formatter(radius_of_curvature[1]))+' m',(330,150), font, 1.2,(255,255,255),2,cv2.LINE_AA)

    # Writing off-center
    cv2.putText(result,'Off-Center:',(740,100), font, 1.2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,str(float_formatter(line_base_pos))+' m',(1000,100), font, 1.2,(255,255,255),2,cv2.LINE_AA)

    return result