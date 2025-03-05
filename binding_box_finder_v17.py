import numpy as np
import cv2
from skimage import filters
import pandas as pd

def blob_maker(time_frames_g, time_frames_r, blob_sigma):
	"""
    Generates a binary image highlighting blob-like structures based on the maximum intensity 
    projection (MIP) of input time-lapse image stacks.

    Parameters:
    ----------
    time_frames_g : numpy.ndarray
        A 3D NumPy array (T, H, W) representing a sequence of grayscale images over time 
        in the green channel, where T is the number of frames, and H, W are image dimensions.
    
    time_frames_r : numpy.ndarray
        A 3D NumPy array (T, H, W) representing a sequence of grayscale images over time 
        in the red channel, where T is the number of frames, and H, W are image dimensions.
    
    blob_sigma : float
        The standard deviation (sigma) for Gaussian blurring. Controls the level of smoothing 
        applied to the image before thresholding.

    Returns:
    -------
    bit8 : numpy.ndarray
        A binary (8-bit) image where detected blobs are highlighted. The shape of the output 
        matches the spatial dimensions (H, W) of the input images.

    """
	
    # Compute the Maximum Intensity Projection (MIP) for both channels over time
	MIP_images_g = np.max(time_frames_g, axis=0)
	MIP_images_r = np.max(time_frames_r, axis=0)
	# Combine MIP images from both channels and compute the final MIP.
	MIPs = [MIP_images_g, MIP_images_r]
	MIP_of_MIPs = np.max(MIPs, axis=0)

    # Apply Gaussian blurring to the maximum intensity projection image
	blurred = filters.gaussian(MIP_of_MIPs, sigma=blob_sigma)  # The sigma value controls the level of smoothing.

    # Compute the Otsu threshold for automatic segmentation.
	binary_image = filters.threshold_otsu(blurred)

	# Create a binary mask where pixels above the threshold are set to True.
	thresh_image = blurred > binary_image
	
	# Convert the binary mask to an 8-bit format (0 and 1 mapped to 0 and 255).
	bit8 = np.uint8(thresh_image)
    
	return bit8
	
def blob_maker_DIC(time_frames_g, blob_sigma):
	"""
    Generates a binary image highlighting blob-like structures based on the maximum intensity 
    projection (MIP) of input time-lapse image stacks.

    Parameters:
    ----------
    time_frames_g : numpy.ndarray
        A 3D NumPy array (T, H, W) representing a sequence of grayscale images over time ,
		where T is the number of frames, and H, W are image dimensions.
    
    blob_sigma : float
        The standard deviation (sigma) for Gaussian blurring. Controls the level of smoothing 
        applied to the image before thresholding.

    Returns:
    -------
    bit8 : numpy.ndarray
        A binary (8-bit) image where detected blobs are highlighted. The shape of the output 
        matches the spatial dimensions (H, W) of the input images.

    """
	
    # Compute the maximum intensity projection across all images
	MIP_images_g = np.max(time_frames_g, axis=0)
	
    # Apply Gaussian blurring to the maximum intensity projection image
	blurred = filters.gaussian(MIP_images_g, sigma=blob_sigma)  # The sigma value controls the level of smoothing.

    # Apply thresholding to obtain a binary image
	binary_image = filters.threshold_otsu(blurred)

	# Create a binary mask where pixels above the threshold are set to True.
	thresh_image = blurred > binary_image

	# Convert the binary mask to an 8-bit format (0 and 1 mapped to 0 and 255).
	bit8 = np.uint8(thresh_image)
    
	return bit8

def reshape_box(x, y, w, h, image_width, image_height, k_resize=50):
	"""
    Expands a bounding box by a fixed amount while ensuring it remains within image boundaries.

    Parameters:
    ----------
    x : int
        The x-coordinate (top-left) of the original bounding box.
    y : int
        The y-coordinate (top-left) of the original bounding box.
    w : int
        The width of the original bounding box.
    h : int
        The height of the original bounding box.
    image_width : int
        The width of the image.
    image_height : int
        The height of the image.
    k_resize : int, optional (default=50)
        The amount by which the width and height of the bounding box should be increased.

    Returns:
    -------
    new_x : int
        The adjusted x-coordinate (top-left) of the reshaped bounding box.
    new_y : int
        The adjusted y-coordinate (top-left) of the reshaped bounding box.
    new_w : int
        The new width of the bounding box.
    new_h : int
        The new height of the bounding box.
    """
	
    # Calculate the center coordinates of the original bounding box
    center_x = x + w / 2
    center_y = y + h / 2

    # Increase width and height by k_resize
    new_w = w + k_resize
    new_h = h + k_resize

    # Compute the new top-left coordinates while keeping the center the same
    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2

   # Ensure the new bounding box does not exceed image boundaries
    if new_x < 0:
        new_x = 0
    elif new_x + new_w > image_width:
        new_x = image_width - new_w

    if new_y < 0:
        new_y = 0
    elif new_y + new_h > image_height:
        new_y = image_height - new_h

    return int(new_x), int(new_y), int(new_w), int(new_h)

def binding_box_find_and_display(bit8, time_frames_green, k_resize=50):
	"""
    Detects contours in a binary image, creates bounding boxes around them, resizes the boxes,
    and overlays them on the first frame of the green channel image.

    Parameters:
    ----------
    bit8 : numpy.ndarray
        A binary (8-bit) image where objects of interest are highlighted.
    
    time_frames_green : numpy.ndarray
        A 3D NumPy array (T, H, W) representing a sequence of grayscale images over time 
        in the green channel, where T is the number of frames, and H, W are image dimensions.
    
    k_resize : int, optional (default=50)
        The amount by which the bounding boxes should be expanded.

    Returns:
    -------
    coord_list : list of lists
        A list containing bounding box coordinates in the format [x, y, w, h].
    
    image_to_display_1 : numpy.ndarray
        The first frame of the green channel image with bounding boxes overlaid.
    """

    # Find contours in the binary image
	contours = cv2.findContours(bit8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]

    # Create coordinates for bounding boxes around the contours
	coord_list = []
	image_height, image_width   = bit8.shape  
	for i in contours:
		# Compute bounding box for each contour
		x, y, w, h = cv2.boundingRect(i)
		
		# Resize the bounding box while keeping it within image bounds
		x, y, w, h = reshape_box(x, y, w, h, image_width, image_height, k_resize)
		
		# Store the modified bounding box coordinates
		coord_list.append([x, y, w, h])

    # Normalize the first frame of the green channel to an 8-bit grayscale image
	normalized_eg_frame = cv2.normalize(time_frames_green[0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	# Make a copy to draw bounding boxes
	image_to_display_1 = normalized_eg_frame.copy()
	
	# Draw the bounding boxes and annotate them with numbers
	for n_coord, (x, y, w, h) in enumerate(coord_list, start=0):
        # Draw a white rectangle around each detected object
		cv2.rectangle(image_to_display_1, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # Add text with coordinate information
		cv2.putText(image_to_display_1, f"({n_coord})", (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the image with the bounding boxes
	cv2.imshow("Boxes", image_to_display_1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

     # Return the list of bounding box coordinates and the annotated image
	return coord_list, image_to_display_1
	
def binding_box_find_and_display_no_reshape(bit8, time_frames_green):
	"""
    Detects contours in a binary image, creates bounding boxes around them, and overlays them 
    on the first frame of the green channel image. Unlike `binding_box_find_and_display`, this function
    does not resize the bounding boxes.

    Parameters:
    ----------
    bit8 : numpy.ndarray
        A binary (8-bit) image where objects of interest are highlighted.
    
    time_frames_green : numpy.ndarray
        A 3D NumPy array (T, H, W) representing a sequence of grayscale images over time 
        in the green channel, where T is the number of frames, and H, W are image dimensions.

    Returns:
    -------
    coord_list : list of lists
        A list containing bounding box coordinates in the format [x, y, w, h].
    
    image_to_display_1 : numpy.ndarray
        The first frame of the green channel image with bounding boxes overlaid.
    """

    # Find contours in the binary image
	contours = cv2.findContours(bit8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]

    # Create coordinates for bounding boxes around the contours
	coord_list = []
	image_width, image_height  = bit8.shape  
	for i in contours:
		x, y, w, h = cv2.boundingRect(i)
		coord_list.append([x, y, w, h])

    # Normalize the first frame of the green channel to an 8-bit grayscale image
	normalized_eg_frame = cv2.normalize(time_frames_green[0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	image_to_display_1 = normalized_eg_frame.copy()

	for n_coord, (x, y, w, h) in enumerate(coord_list, start=0):
        # Draw a rectangle around each bounding box
		cv2.rectangle(image_to_display_1, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # Add text with coordinate information
		cv2.putText(image_to_display_1, f"({n_coord}", (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the image with the bounding boxes
	cv2.imshow("Boxes", image_to_display_1)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

    # Return the list of coordinates
	return coord_list, image_to_display_1



def draw_boxes_from_coordinates(coordinates_list, image_of_worms, filename_boxes, starting_worm):
	
	"""
    Draw bounding boxes around objects in an image and display it.

    Parameters:
    - coordinates_list (list): A list of tuples containing (x, y, w, h) coordinates for each bounding box.
    - image_of_worms (numpy.ndarray): The input image containing objects.
	- filename_boxes (path): The path with directory and name

    Displays the image with bounding boxes and coordinate information.

    Example:
    draw_boxes_from_coordinates([(10, 20, 30, 40), (50, 60, 70, 80)], image_of_worms)
    """
	
	#normalize the imput image
	normalized_image = cv2.normalize(image_of_worms, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	image_to_display = normalized_image.copy()
	
	df = pd.DataFrame(coordinates_list, columns=['x', 'y', 'w', 'h'])
	
	# Sort primarily by y (descending) and then by x (ascending) within each row
	df_sorted = df.sort_values(by=['y', 'x'], ascending=[False, True]).reset_index(drop=True)
	
	# Convert back to list of tuples with box and number
	numbered_boxes = df_sorted[['x', 'y', 'w', 'h']].values.tolist()

		# Iterate through the list of coordinates
	for n_coord, (x, y, w, h) in enumerate(numbered_boxes, start=starting_worm):
		# Draw a rectangle around each bounding box
		cv2.rectangle(image_to_display, (x, y), (x+w, y+h), (255, 255, 255), 2)
		
		# Add text with coordinate information
		cv2.putText(image_to_display, f"({n_coord}", (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		# Convert boxes to a DataFrame for sorting

	# Display the image with the bounding boxes
	cv2.imshow("Boxes", image_to_display)
	#Save image
	cv2.imwrite(filename_boxes, image_to_display)
	# Wait for a key press and close the window
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return image_to_display


