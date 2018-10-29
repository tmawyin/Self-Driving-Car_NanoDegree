## Udacity Self Driving Car Project 2 - Advance Lane Finding
Author: Tomas Mawyin
 
#### The objective of this project is to investigate advanced computer vision techniques to detect lane lines on the road. The techniques used in this project will be used to detect lanes on static images and videos in order to simulate road conditions. 

---

**Goals and Steps of the project:**

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/find_corners.jpg "Corners"

[image2]: ./writeup_images/undistort.jpg "Undistort"
[image3]: ./writeup_images/test_undistort.jpg "Road Transformed"

[image4]: ./writeup_images/sobelx.jpg "Sobel X"
[image5]: ./writeup_images/hls.jpg "HLS"
[image6]: ./writeup_images/threshold.jpg "Binary Example"

[image7]: ./writeup_images/perspective.jpg "Warp Example"

[image8]: ./writeup_images/windowSlide.jpg "Window Sliding"
[image9]: ./writeup_images/searchPoly.jpg "Polynomial Search"
[image10]: ./writeup_images/findCentroid.jpg "Centroid Search"

[image11]: ./writeup_images/finalResult.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

#### The purpose of this document is to describe the steps that have been implemented for the advanced lane detection. I will provide examples and code snippets in each of the steps to show how the pipeline works.

### Camera Calibration

#### Objective: Compute camera matrix and distortion coefficients to undistort images

The first step in the pipeline is to find the camera matrix and the distortion coefficients in order to undistort images. This is done to avoid any disturbance that arise by taking pictures or videos with a camera. Typically, cameras distort images at the edges as the light bends because of the use of lenses. By implementing this step we remove any unnecessary effects and "flatten" all edges in the image.

We make use of twenty chessboard images taken at different locations and angles. First, we find all "corners" of the chessboard image; this means finding the (x,y) coordinates of the internal points of the chessboard, we call such points the image points. These points are then mapped to object points representing the ideal grid for the chessboard (x,y) coordinates. To help in the coding process, I generate a file called `ald.py` (Advanced Lane Detection) to hold all helper functions. All the camera calibration code can be found in this file from lines 10 to 61.

To find all corners and obtain the image and object points I created a function called `find_corners()`. The function takes a chessboard image, uses OpenCV to turn it into a gray scale and using the function `cv2.findChessboardCorners()` we obtain the x and y locations of all the corners. An example of what the function returns is shown here:

![alt text][image1]

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients. I made use of a function called `undistort_image()` that uses the `cv2.calibrateCamera()` function from OpenCV.  I applied this distortion correction to the test image using the `cv2.undistort()` function. The figure below shows a comparison between the distorted image (left) and the undistorted image (right).

![alt text][image2] 

### Pipeline (single images)

#### Step 1. Image correction

Now that we have the object and image points, it becomes easier to demonstrate the correction on the test images (or each frame of the video). The correction applied is shown in the image below but the correction is very subtle. The figure shows the comparison between the distorted and undistorted image.

![alt text][image3]

#### Step 2. Gradients and Color Spaces

The next step of the pipeline is to find a way to highlight the lane lines. This is done by a combination of color spaces and/or by applying different gradient operations to the image. In order to perform this action, I made use of five (5) helper functions that can be found in the `ald.py` file (See lines 64 to 229). These functions are:

* `abs_sobel_thresh()` is a function that given the orientation (x or y) performs the Sobel operator or takes the gradient of the image in the given direction. This function returns a binary images (black & white) that keeps only the pixels that are between a given threshold. The advantage of this function is that by providing a direction we can try to maintain mainly horizontal or vertical lines. An example of this function is shown here:

![alt text][image4]

* `mag_thresh()` is the next helper function that calculates the magnitude of the Sobel operator. This is done by taking applying the formula: `np.sqrt(np.square(sobelx)+np.square(sobely))` where "sobelx" and "sobely" are the Sobel operators taken in the x and y directions respectively. The function returns a binary image that maintains sobel magnitudes between an specified threshold.
* `dir_threshold()` is a function that calculates the direction of the Sobel operators, think of this as the angle from a polar coordinate of a point. The direction is calculated by taking the inverse tangent of the "sobely" over the "sobelx" variables. Similar to the other functions, this one returns a binary function that holds values in between the given threshold.
* `hls_select()` is a function that takes a channel (either "h", "l", or "s") and computes the HLS selection. It returns a binary image with threshold applied. The HLS color space helps in the detection of color lines so that we can detect wither yellow or white lines. An example of this is the image below, where the "s" channel was selected to have a good detection of the yellow line on the left side.

![alt text][image5]

* `hsv_select()` is a function similar to the one above. This function instead selects between the H,S, and V channels of the image and performs a similar thresholding as the previous functions.

The helper functions above are used as a method to control the color scales and gradients of the image in order to obtain a better binary image which contains the lanes. In reality, none of the images by itself can provide a good solution for this and therefore we need to combine one or more of these helper functions. To do this, I created a new function called `threshold_find()`, the function is as follows:

```phython
def threshold_find(img)
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
```

The code above shows the combination of the helper functions. Note that we can combine different color spaces to distinguish between yellow and white lines. Similarly, we use some of the gradient functions to eliminate any undesired pixels near the lanes. Note that each of the functions above contains its own threshold range in order to satisfy the lane detection method. The ideal combined image is shown below as compared to the original image.  

![alt text][image6]

#### Step 3. Perspective Transform

The next step in the pipeline is to perform a perspective transformation. This is done to view the lane lines from a different perspective (birds-eye view) and be able to detect the lanes in an easier way. I make use of a function called `perspective_transform()` in the `ald.py` file (see lines 232 to 245 for reference). The function takes the image, as well as source (`src`) and destination (`dst`) points. We make use of the OpenCV functions `cv2.getPerspectiveTransform(src, dst)`  to generate a matrix that maps the source to the destinations points. Similarly, the function `cv2.warpPerspective(img, M, img_size, flags = cv2.INTER_LINEAR)` is used to apply such matrix to the original image and obtain the new transformed image. 

To choose the source and destination points I made use of the image size and a few offset values (found via trial and error) as follows:

```python
# Defining the source points
b_offset = 450
t_offset = 50
v_offset = 92
src = np.float32([[img_x/2-b_offset, img_y], 
		[img_x/2-t_offset, img_y/2+v_offset], 
		[img_x/2+t_offset, img_y/2+v_offset], 
		[img_x/2+b_offset, img_y]])
# Defining the destination points
dst = np.float32([[200,img_y], [200,0], [1000,0], [1000, img_y]])
```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The image on the left is the bird-eye view showing the parallel lanes. However, to detect the lines better, the code will actually use the binary image (with the thresholds as shown in Step 2).

![alt text][image7]

#### Step 4. Identify lane lines

Now that we have the perspective transformation, the lane lines can be identified. This is where the bulk of the code resides and in the file `ald.py` lines 248 to 455 show all the functions that are used to perform this task. There are three main techniques used to detect lane lines used in the code:

1. `find_polynomial()` and `find_lane_pixels()` work together as a Slidng Window technique for detecting lane lines. The main function is the latter and it takes a binary image as input and provides the resultant image with the left and right fitted lines. The function performs the following way:
	* The function starts by taking a binary image and finding a histogram of pixels on the bottom half of the image. The histagram is calculated by adding all the pixel values along the y-axis, finding the peak value, and use this as a starting point for the lane detection in the x-axis. Note that this is done in the left and right side of the image.
	* The next step is to set up a sliding window by defining some margins. Similarly, we define how many windows or layers to use in the image. The next step is to set up a loop to go over each window and identify the active pixels within each window.
	* Once the active pixels are identify we append their location to a variable that will later be used to fit a second order polynomial to. We will use the polynomial coefficients in the next technique, so the code keeps track of these coefficients.
	* Finally, we move the window around to the next layer based on where the active pixels were found. The picture shown below, is a representation of how this window sliding technique and how the lines are detected with the polyfit.

![alt text][image8]

2. `search_around_poly()` is the next technique. As the name implies, this function searches around the previous polynomial fit to find the active pixels in a binary image. The function takes the binary image and the left and right polynomial coefficients as inputs and returns the resultant image and the respective left and right polynomial points. The function performs the following way:
	* We set up a margin that will be used to help us find near the polyfitted line. Imagine searching the previous fitted line +/- some margin. This is possible since we expect the lane in the current fram to be a continuation from the lane in the previous frame.
	* We check for activate pixels near the search area (including the margin) and we keep a list of all these points. Note that this operation can be done very quickly since we are searching the entire image at once for those active pixels near the fitted line. Here is the main code for this function, and as you can see there is a simple search around the fitted line +/- the margin
	```python
	# Setting the area of search based on activated x-values within the +/- margin of our polynomial function
    left_lane_inds = ((nonzerox > (np.poly1d(left_fit)(nonzeroy) - margin)) & (nonzerox < (np.poly1d(left_fit)(nonzeroy) + margin)))
    right_lane_inds = ((nonzerox > (np.poly1d(right_fit)(nonzeroy) - margin)) & (nonzerox < (np.poly1d(right_fit)(nonzeroy) + margin)))
    ```
	* Once the new pixels and their location are found, we can fit a new second order polynomial and obtain the lane lines from the new frame. To visualize this, the image below shows the margin and the fitted line.

![alt text][image9]

3. `find_window_centroid()` is our last technique. It should be pointed out that this technique is a stand alone techinque that can be used instead the first technique shown here. The function takes a binary image as input and returns the resultant image and the fitted lane points for the left and right sides. This function is very similar to the sliding window technique but with some modifications. The main difference is that this function uses what is known as "convolution" to try to find the centroids of the lanes. The following steps are used:
	* A window (width and height) is set up. This window is used as one function containing active pixels (of value = 1). The first operation is to find an initial x-value point to start the search. This is accomplished by finding the convolution between the window of active pixels and the left and right sides of the image.
	* Once the starting points are found, we continue using the convolution function (window with active pixel values) across different layers of the image. Imagine going up the image and finding the location of the most active pixel. These location points are then appended to a centroid list and are used to fit a polynomial.
	* The final step of this function is to combine the centroid values and find in the image where all the actual lane pixels are. This is done with the help of a masking function `window_mask()`. Finally, we use these points to get the second order polynomial and return the left and right side points. Here is an image of how the search looks like:

![alt text][image10]

The code is set up in such a way that techniques #1 or #3 are use at the start of the code (when there are no found polynomial coefficients yet). Later iteration of the code use technique #2 to improve on the efficiency of the code. Note that the main code is set up as a pipeline function inside a class called `Line()`. This is done to keep track of required values and later smooth some of the data over some number of frames. All the functions mentioned above and the functions that follow are part of the `pipeline()` function inside the class that takes on an image and computes the final result that you will see in Step 6 of this report.

#### Step 5. Radius of curvature and Off-center vehicle position

Now that the lanes (polynomial fit) have been detected, it is time to compute the curvature of the road. This is done in lines 548 through 575 in the `ald.py` file. This part of the code is very straight forward as it is a mathematical formula applied to the left and right sides of the road. I will let the code show the implementation.

```python
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
```

There is a few things to point out, first is that the code uses a conversion from pixels to meters in the x and y directions. The parameters are set up based on the lane lines and the pixels used in the image. The function takes the left and right polynomial coefficients, that are used to calculate the curvature by evaluating the `y_eval` point.  Finally, the off-center distance of the vehicle is calculated by assuming the middle of the image (in x-direction) is the center of where the camera is mounted.  

#### Step 6. Resultant image

At the end of the pipeline we make use of two important functions (see lines 578 to 618 in the `ald.py` file). The first one is called `unwrap_image()` and is used to achive two things: 1) to mark the lane lines and paint between them and 2) reverse the perspective transfer. This function makes use of the perspective transform step but reversing the source and destination points. The second function is `write_image()` and it is used to write text on an image. In particular the curvature and the off-center distance. This function returns the image with the text written at the top of the image.

As stated before, the main portion of the code found in the `Project_2.py` file. This code is developed around a class called `Line()`. The class contains variables that are needed to keep track of certain values and it contains one function to execute the pipeline mentioned throughout this report. The resultant of this function is an image with the lanes marked and the curvature written at the top of the image. The figure below shows a comparison between the original and resultant image.
 
![alt text][image11]

---

### Pipeline (video)

The pipeline was constructed to run on images or videos. The way to run the code for a video is the generate an instance of the `Line()` class and pass the 	`pipeline()` function to the MoviePy videoClip class. The following code does exactly this and the resultant video is attached:

```python
# ----- TESTING VIDEO
lanes = Line(obj_points, img_points) #<- Here is an instance of the Line() class
white_output = 'output_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(lanes.pipeline) # <- We pass the pipeline() function here
white_clip.write_videofile(white_output, audio=False)
```

Here's a [link to my video result](./writeup_images/output_video.mp4)

---

### Discussion

Implementing this code was very fun and it was very exciting to learn about the computer vision techniques to detect lane markings on the road. 

The major problem found is that this only work for well distinguished lines in the road. For road without well painted lines, this code will not be robust enough and other techniques might need to be investigated. Additionally, the efficency of the code can decrease if more techniques are implemented. This can be avoided and easily be changed by including only techniques that are appropriate. Another implementation code that can be included to improve on the code is to make use of Machine Learning techniques to learn about different road scenarios and make the code learn how to handle those.

Other aspects that can be worked throught are the calibration required when using different color spaces. For this application, I tried to be as effective as possible for the images and videos (very limited use cases) but there should be a more robust implementation that probably can measure and be more dynamic in apply different color spaces and gradients. It would be ideal to learn more about how to deal with images and use computer vision techniques to work through the noise in the images/videos.

On a personal level, some of the difficulties that I found was to implement the class to keep track of the needed parameters. I would like to learn more about Python class implementation to improve the code and make it more robust, smooth, and implement additional safety checks. 
