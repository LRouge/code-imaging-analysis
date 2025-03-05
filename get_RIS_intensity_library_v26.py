import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, label, labeled_comprehension
from skimage.filters import threshold_otsu, threshold_yen
import matplotlib.pyplot as plt  # Plotting library
import cv2
import gc

def get_single_worm(coord, image_green, image_red):
	"""
	Extracts a region of interest (ROI) from the input images based on the provided coordinates.

	Parameters:
	- coord (tuple): Tuple containing (x, y, height, width) of the region to extract.
	- image_green (numpy.ndarray): The green channel image.
	- image_red (numpy.ndarray): The red channel image.

	Returns:
	- tuple: A tuple containing the extracted green and red channel images.
	"""
	# Unpack coordinates for the bounding box
	x, y, h, w = coord
	
	# Extract the worm region from both green and red channel images
	worm_green = image_green[:, y:(y+h), x:(x+w)] # Cropping for green channel
	worm_red = image_red[:, y:(y+h), x:(x+w)]  # Cropping for red channel

	# Delete original images to free up memory (optional, useful for large datasets)
	del image_green
	del image_red
	gc.collect()

	# Return the extracted regions
	return worm_green, worm_red

    
def get_thresholded_image_yan(worm_red_frame, worm_green_frame, k_blur, k_blur_analysis):\
	"""
    Applies Gaussian blurring and Yen's thresholding to the red and green channel images 
    to generate thresholded binary masks.

    Parameters:
    - worm_red_frame (numpy.ndarray): The red channel image (single frame).
    - worm_green_frame (numpy.ndarray): The green channel image (single frame).
    - k_blur (float): Sigma value for Gaussian blur applied to both red and green images.
    - k_blur_analysis (float): Additional Gaussian blur applied to the red channel for further analysis.

    Returns:
    - tuple of numpy.ndarray: 
        - Binary mask of the green channel after thresholding.
        - Binary mask of the red channel after thresholding.
        - Binary mask of the red channel with additional blur applied before thresholding.
    """

	# Gaussian blur
	worm_green_gb = gaussian_filter(worm_green_frame, sigma=k_blur)
	worm_red_gb = gaussian_filter(worm_red_frame, sigma=k_blur)
	worm_red_gb_analysis = gaussian_filter(worm_red_frame, sigma=k_blur_analysis)

	# Threshold using Yan's method
	thresh_value_worm_g = threshold_yen(worm_green_gb)
	thresh_value_worm_r = threshold_yen(worm_red_gb)
	thresh_value_worm_r_analysis = threshold_yen(worm_red_gb_analysis)

	# Apply thresholding to obtain binary masks
	worm_green_to_np = worm_green_gb > thresh_value_worm_g
	worm_red_to_np = worm_red_gb > thresh_value_worm_r
	worm_red_to_np_analysis = worm_red_gb_analysis > thresh_value_worm_r_analysis

	# Convert boolean masks to 8-bit images (0 or 255)
	worm_green_to_np = worm_green_to_np.astype(np.uint8) * 255
	worm_red_to_np = worm_red_to_np.astype(np.uint8) * 255
	worm_red_to_np_analysis = worm_red_to_np_analysis.astype(np.uint8) * 255


	return worm_green_to_np, worm_red_to_np, worm_red_to_np_analysis

def get_RIS_intensities(worm_green_to_np, worm_red_to_np, worm_red_to_np_analysis,  worm_green_frame, worm_red_frame):
	"""
    Calculate the mean intensities of the regions of interest (ROIs) in the red and green channel images, 
    based on the contours detected in the thresholded binary masks.
    
    Parameters:
    - worm_green_to_np (numpy.ndarray): Thresholded binary image of the green channel.
    - worm_red_to_np (numpy.ndarray): Thresholded binary image of the red channel.
    - worm_red_to_np_analysis (numpy.ndarray): Thresholded binary image of the red channel for analysis.
    - worm_green_frame (numpy.ndarray): Original green channel image.
    - worm_red_frame (numpy.ndarray): Original red channel image.
    
    Returns:
    - tuple: Mean intensities for the green channel (GCamp), the green channel uncorrected, 
             and the red channel (mKate), along with the corrected centroid coordinates for the red channel.
    """

	# Find contours in the thresholded binary images for each channel
	cnts_red, _ = cv2.findContours(worm_red_to_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts_green, _ = cv2.findContours(worm_green_to_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts_red_analysis, _ = cv2.findContours(worm_red_to_np_analysis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# Check if contours exist in all channels before proceeding
	if len(cnts_red)>0 and len(cnts_green)>0 and len(cnts_red_analysis) > 0:

		# Sort contours to find the largest one in each channel
		biggest_cnt_red = sorted(cnts_red, key=cv2.contourArea, reverse=True)[0]
		biggest_cnt_red_analysis = sorted(cnts_red_analysis, key=cv2.contourArea, reverse=True)[0]
		biggest_cnt_green = sorted(cnts_green, key=cv2.contourArea, reverse=True)[0]
	
		# Get the moments to calculate the centroid
		M_red = cv2.moments(biggest_cnt_red)
		M_green = cv2.moments(biggest_cnt_green)
		
		# Calculate centroid coordinates from the moments
		cx_r = int(M_red['m10'] / M_red['m00'])
		cy_r = int(M_red['m01'] / M_red['m00'])
		
		cx_g = int(M_green['m10'] / M_green['m00'])
		cy_g = int(M_green['m01'] / M_green['m00'])
	
		# Calculate the movement vector between centroids of the red and green contours
		movement_vector = (cx_g - cx_r , cy_g - cy_r)

		# Correct the coordinates of the red contour based on the movement vector
		corrected_contour = biggest_cnt_red
		for i in range(len(corrected_contour)):
		    # Correct the coordinates by adding the movement vector
		    x, y = corrected_contour[i][0]
		    corrected_x = x + movement_vector[0]
		    corrected_y = y + movement_vector[1]
		
		    # Update the coordinates of the current point in the contour
		    corrected_contour[i][0] = [corrected_x, corrected_y]
	
	
		# Create masks for the contours using filled areas
		mask_green = np.zeros_like(worm_green_frame, dtype=np.uint8)
		cv2.drawContours(mask_green, [corrected_contour], -1, (255), thickness=cv2.FILLED)
	
		mask_red = np.zeros_like(worm_red_frame, dtype=np.uint8)
		cv2.drawContours(mask_red, [biggest_cnt_red_analysis], -1, (255), thickness=cv2.FILLED)
		
	
		# Label the connected regions in the masks
		labeled_array_red_to_red, num_labels_red = label(mask_red)
		labeled_array_red_to_green, num_labels_green = label(mask_green)
		labeled_array_green_to_green, num_labels_g = label(worm_green_to_np)
		
		# Calculate mean intensity for each labeled region
		mean_intensity_gcamp = labeled_comprehension(worm_green_frame, labeled_array_red_to_green, np.arange(1, num_labels_green + 1), np.mean, float, 0)[0]
		mean_intensity_gcamp_uncorrected = labeled_comprehension(worm_green_frame, labeled_array_green_to_green, np.arange(1, num_labels_g + 1), np.mean, float, 0)[0]
		mean_intensity_mkate = labeled_comprehension(worm_red_frame, labeled_array_red_to_red, np.arange(1, num_labels_red + 1), np.mean, float, 0)[0]

	else:
		# Return "NaN" values if contours are not found
		mean_intensity_gcamp = "NaN"
		mean_intensity_gcamp_uncorrected = "NaN"
		mean_intensity_mkate = "NaN"
		cx_r = "NaN"
		cy_r = "NaN"
	
	return mean_intensity_gcamp, mean_intensity_gcamp_uncorrected, mean_intensity_mkate, cx_r, cy_r



def single_frame_analysis(timepoint, worm_green, worm_red, k_blur, k_blur_analysis, n_worm, replicate):
	"""
    Analyzes a single frame of worm images, extracting features and storing them in a dictionary.
    
    Parameters:
    - timepoint (int): Timepoint of the frame in the sequence.
    - worm_green (numpy.ndarray): 3D array of green channel images (time, height, width).
    - worm_red (numpy.ndarray): 3D array of red channel images (time, height, width).
    - k_blur (float): Sigma value for Gaussian blur.
    - k_blur_analysis (float): Sigma value for Gaussian blur used during analysis.
    - n_worm (int): Index of the worm.
    - replicate (int): Replicate number for the experiment.

    Returns:
    - dict: A dictionary containing the analyzed features, including intensities, centroid coordinates, and metadata.
    """
	# Extract the green and red channel frames for the specified timepoint
	worm_green_frame = worm_green[timepoint]
	worm_red_frame = worm_red[timepoint]

	 # Apply thresholding and Gaussian blurring to the images
	worm_green_to_np, worm_red_to_np, worm_red_to_np_analysis  = get_thresholded_image_yan(worm_red_frame, worm_green_frame, k_blur, k_blur_analysis)

	# Calculate the mean intensities and centroids based on contours found in the thresholded images
	mean_intensity_gcamp, mean_intensity_gcamp_uncorrected, mean_intensity_mkate, cx_r, cy_r = get_RIS_intensities(worm_green_to_np, worm_red_to_np, worm_red_to_np_analysis,  worm_green_frame, worm_red_frame)

	# Create a dictionary to store the analyzed features
	worm_dict = {'time': timepoint, # Timepoint of the frame
				 'n_worm': n_worm, # Index of the worm 
				 'intensity_gcamp': mean_intensity_gcamp,  # GCamp mean intensity
				 'intensity_gcamp_uncorrected': mean_intensity_gcamp_uncorrected, # GCamp uncorrected intensity
				 'intensity_mKate': mean_intensity_mkate,  # mKate mean intensity
				 'centroid_x': cx_r, # x-coordinate of the red contour centroid
				 'centroid_y': cy_r, # y-coordinate of the red contour centroid
				 'replicate': replicate # Replicate number for the experiment
				}

	return worm_dict # Return the dictionary containing the analyzed features

def over_time_analysis(timerange, worm_green, worm_red, k_blur, k_blur_analysis, n_worm, replicate):
    """
    Analyzes worm images over a sequence of timepoints.

    Parameters:
    - timerange (list): List of timepoints to analyze.
    - worm_green (numpy.ndarray): 3D array of green channel images (time, height, width).
    - worm_red (numpy.ndarray): 3D array of red channel images (time, height, width).
    - k_blur (float): Sigma value for Gaussian blur.
    - k_blur_analysis (float): Sigma value for Gaussian blur during analysis.
    - n_worm (int): Index of the worm.
    - replicate (int): Replicate number for the experiment.

    Returns:
    - list: A list of dictionaries containing the analyzed features for each timepoint.
    """
	# Initialize an empty list to store the results for each timepoint
	worm_dict_list = []

	# Loop through each timepoint in the provided timerange
	for timepoint in timerange:
		worm_dict = single_frame_analysis(timepoint, worm_green, worm_red, k_blur, k_blur_analysis, n_worm, replicate)
		# Append the results to the list
		worm_dict_list.append(worm_dict)

	return worm_dict_list # Return the list of dictionaries containing analyzed features for each timepoint

def multiple_worm_analysis(coord, n_worm, timerange, images_gcamp, images_mKate, k_blur, k_blur_analysis, replicate):
	"""
    Analyzes multiple worms over a sequence of timepoints.

    Parameters:
    - coord (tuple): Coordinates of the worm to analyze in the form (x, y, height, width).
    - n_worm (int): Index of the worm.
    - timerange (list): List of timepoints to analyze.
    - images_gcamp (numpy.ndarray): 3D array of green channel images (time, height, width).
    - images_mKate (numpy.ndarray): 3D array of red channel images (time, height, width).
    - k_blur (float): Sigma value for Gaussian blur.
    - k_blur_analysis (float): Sigma value for Gaussian blur during analysis.
    - replicate (int): Replicate number for the experiment.

    Returns:
    - list: A list of dictionaries containing the analyzed features for each timepoint for the specified worm.
    """
	# Initialize an empty list to store the analysis results
	worm_dict_list = []
	
	# Extract the region of interest (ROI) for the specified worm based on the provided coordinates
	worm_green, worm_red = get_single_worm(coord, images_gcamp, images_mKate)

	# Perform over-time analysis for the specified worm over the list of timepoints
    # Extend the worm_dict_list with results from over_time_analysis
	worm_dict_list.extend(over_time_analysis(timerange, worm_green, worm_red, k_blur, k_blur_analysis, n_worm, replicate))
	# Return the list of dictionaries containing analyzed features for the specified worm
	return worm_dict_list
