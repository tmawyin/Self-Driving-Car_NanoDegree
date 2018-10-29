# PROJECT 2 - ADVANCED LANE DETECTION
# AUTHOR - TOMAS MAWYIN

# Objective: Generate a pipeline to detect lane lines on a video. Initial testing to be done in static images

# Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import cv2 # Open-CV library
from moviepy.editor import VideoFileClip
from ald import *

class Line():
	def __init__(self, obj_points, img_points):
		# Adding object and image points for undistortion
		self.obj_points = obj_points
		self.img_points = img_points
		# Was the line detected in the last iteration?
		self.detected = False
		# Polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]  
		# Radius of curvature of the line in meters
		self.radius_of_curvature = None 
		# Distance in meters of vehicle center from the line
		self.line_base_pos = None 

		#x values of the last n fits of the line
		self.recent_xfitted = [] 
		#average x values of the fitted line over the last n iterations
		self.bestx = None     
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = []  
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float') 
		#x values for detected line pixels
		self.allx = None  
		#y values for detected line pixels
		self.ally = None

	def pipeline(self, img, debug=False):
		# ----- UNDISTORT
		# Step 1 - Undistorting the image
		undist_img = undistort_image(img, self.obj_points, self.img_points)

		# ----- THRESHOLDING
		# Step 2 - Thresholding the image
		combined = threshold_find(undist_img)

		# ----- PERSPECTIVE TRANSFORM FUNCTIONS
		# Getting image size
		img_y = undist_img.shape[0]
		img_x = undist_img.shape[1]

		# Defining the source points
		b_offset = 450
		t_offset = 50
		v_offset = 92
		src = np.float32([[img_x/2-b_offset,img_y], [img_x/2-t_offset,img_y/2+v_offset], 
			[img_x/2+t_offset,img_y/2+v_offset], [img_x/2+b_offset,img_y]])
		# Defining the destination points
		dst = np.float32([[200,img_y], [200,0], [1000,0], [1000, img_y]])
		# Getting the perspective transform
		warped = perspective_transform(combined, src, dst)

		# ----- HISTOGRAM, LANE FINDING, AND CURVATURE
		for_testing = debug
		if self.detected == False:
			# lane_img, left_fitx, right_fitx, ploty, self.detected = find_polynomial(warped, debug=for_testing)
			lane_img, left_fitx, right_fitx, ploty, self.detected = find_window_centroids(warped,debug=for_testing)
			self.current_fit = [np.polyfit(ploty, left_fitx, 2), np.polyfit(ploty, right_fitx, 2)]
			self.radius_of_curvature, self.line_base_pos = measure_curvature_center(self.current_fit, ploty)
		elif self.detected == True:
			lane_img, left_fitx, right_fitx, ploty, self.detected = search_around_poly(warped, self.current_fit[0], self.current_fit[1], debug=for_testing)
			self.current_fit = [np.polyfit(ploty, left_fitx, 2), np.polyfit(ploty, right_fitx, 2)]
			self.radius_of_curvature, self.line_base_pos = measure_curvature_center(self.current_fit, ploty)

		# Trying to perform some smoothing functionality
		frame = 4
		self.best_fit.append(self.current_fit)
		self.recent_xfitted.append([left_fitx , right_fitx])
		if (len(self.best_fit) == frame):
			self.current_fit = np.average(self.best_fit, axis=0)
			# Averaging some of the x-points for the left and right lanes
			fit_avg = np.average(self.recent_xfitted[-frame:], axis =0)
			left_fitx = fit_avg[0]
			rigth_fitx = fit_avg[1]
			self.best_fit = []

		# ----- UNWARP IMAGE
		# Unwarping image - pass the src and dst points to invert
		newwarp = unwrap_image(warped, left_fitx, right_fitx, ploty, src, dst)
		
		# ----- GENERATING FINAL IMAGE
		# Combine the result with the original image
		result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

		# Writing final results in the image
		result = write_image(result, self.radius_of_curvature, self.line_base_pos)

		return result


# ---------- TESTING ----------

# ----- IMAGE CALIBRATION
# Setting up the list of files and finding corners
files = glob.glob('camera_cal/calibration*.jpg')
# Obtaining the points to calibrate the camera
img_points, obj_points = find_corners(files, nx=9, ny=6, debug=False)

# ----- TESTING IMAGES
all_images = os.listdir("test_images/")
for image in all_images:
	# Opening each of the test images
	image = mpimg.imread('test_images/'+ image)
	lanes = Line(obj_points, img_points)
	result = lanes.pipeline(image)

	# Plotting
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
	f.tight_layout()
	ax1.imshow(image)
	ax2.imshow(result)
	plt.show()

# ----- TESTING VIDEO
lanes = Line(obj_points, img_points)
white_output = 'output_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(lanes.pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

# white_output = 'output_challenge_video.mp4'
# clip1 = VideoFileClip("challenge_video.mp4").subclip(0,5)
# white_clip = clip1.fl_image(lanes.pipeline) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)

