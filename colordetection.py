import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter

def extractSkin(image, segmentation_mask=None):
    # If a segmentation mask is provided, use it
    if segmentation_mask is not None:
        skin = cv2.bitwise_and(image, image, mask=segmentation_mask)
    else:
        # Converting from BGR Color Space to HSV
        img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Defining HSV Thresholds
        lower_threshold = (0, 48, 80)
        upper_threshold = (20, 255, 255)

        # Single Channel mask, denoting the presence of colors in the above threshold
        skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

        # Cleaning up mask using Gaussian Filter
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

        # Extracting skin from the threshold mask
        skin = cv2.bitwise_and(image, image, mask=skinMask)

    # Return the Skin image
    return skin

def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)

def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionary of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation

def extractDominantColor(image, number_of_colors=5, hasThresholding=False):
    
    if hasThresholding:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    color_information = getColorInformation(estimator.labels_, estimator.cluster_centers_, hasThresholding)

    most_dominant_color_info = max(color_information, key=lambda x: x["color_percentage"])

    # Extract only the RGB color values as a NumPy array
    most_dominant_color = np.array(most_dominant_color_info["color"])


    return most_dominant_color.astype('uint8')

def adjust_hsv_dominance(dominant_color, hsv_adjust=0.2):
  """
  Adjusts the HSV color space of a dominant color.

  Args:
    dominant_color: A NumPy array (3,) representing the RGB values of the dominant color.
    hsv_adjust: A float value between 0 and 1 controlling the adjustment intensity.

  Returns:
    A NumPy array (3,) with the adjusted RGB values of the dominant color.
  """

  # Convert to HSV color space
  hsv = cv2.cvtColor(dominant_color.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)
  hsv = hsv[0, 0, :]

  # Adjust saturation slightly
  hsv[1] = min(255, hsv[1] * (1 + hsv_adjust))

  # Adjust hue subtly (optional)
  # hsv[0] = (hsv[0] + hsv_adjust * 360) % 360

  # Convert back to RGB
  rgb = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2RGB)
  rgb = rgb[0, 0, :]

  return rgb.astype('uint8')



