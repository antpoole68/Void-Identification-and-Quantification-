import cv2
import numpy as np
from scipy.signal import medfilt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage import exposure


def read_and_preprocess(image_path):
    # Read in image convert to gray scale and median filter
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    image = medfilt(image)
    # Determine mean intensity
    mean_intensity = np.mean(image)
    if mean_intensity > 120 or mean_intensity < 55:
        print("Error Outlier Image cannot categorize")
        return image
    else:
        return image
    
def histogram_eq(image, openCV=True):
    if (openCV):
        return cv2.equalizeHist(image)
    

def find_image_with_highest_mean_intensity(tensor):
    # Calculate the mean intensity for each of the 78 images
    mean_intensities = np.mean(tensor, axis=(0, 1))

    # Find the index of the image with the highest mean intensity
    max_intensity_image_index = np.argmax(mean_intensities)

    # Extract and return the image with the highest mean intensity
    highest_intensity_image = tensor[:, :, max_intensity_image_index]

    return highest_intensity_image
    

def void_isolation(image, openCV, num_seg, compactness):
    if (openCV):
        original_image = image.shape
        # Reshape the image to a 2d array
        fil_image = image.reshape((-1,))
        fil_image = np.float32(fil_image)
        # Define the criteria
        k = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .1)
        ret,label,center=cv2.kmeans(fil_image,num_seg,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        
        return res.reshape(original_image)
    else:
        k = num_seg  # Adjust the number of clusters as needed
        # Perform k-means clustering
        image_2d = image.reshape(-1, 1)  # Reshape the image to a 2D array
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(image_2d)
        labels = kmeans.predict(image_2d)
        # Reshape the cluster labels back to the original image shape
        segmented_image = labels.reshape(image.shape)
        
        return segmented_image
    


def gabor_filter(image):
    lambdas = []
    thetas = []
    sigmas = []
    ms = [2, 2.25, 2.5 , 2.75, 3]
    for m in ms:
        lambdas.append(3*(2**m))
    for n in range(15):
        thetas.append(12.25*n)
    for f in range(1,76):
        sigmas.append(.15*lambdas[(f-1)%5])
    
    i = 0
    kernels = []
        
    for idx, sig in enumerate(sigmas):
        ksize = 30
        gamma = .5
        psi = 0
        lam = lambdas[idx%5]
        theta = thetas[idx%15]
        kernels.append(cv2.getGaborKernel((ksize, ksize), sig, theta, lam, gamma, psi, ktype=cv2.CV_32F))
        
    # apply the kernels to the images
    images = []
    for kernel in kernels:
        fimg = (cv2.filter2D(image, cv2.CV_32F, kernel))
        images.append(fimg/(fimg.max())*255)

    height, width = image.shape
    # Create mesh grids of x and y values
    x_mesh, y_mesh = np.meshgrid(np.arange(width), np.arange(height))
    images.append(image)
    images.append(x_mesh)
    images.append(y_mesh)
    images = np.array(images)
    images = np.transpose(images, (1, 2, 0))
    
    orginal_shape = images.shape
    
    # Reshape the image to a 2d array
    fil_image = images.reshape((-1,78))
    fil_image = np.float32(fil_image)
    # Define the criteria
    k = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .1)
    ret,label,center=cv2.kmeans(fil_image,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    
    # return the data to to the correct shape
    set_images = res.reshape(orginal_shape)
    highest_intensity_image = find_image_with_highest_mean_intensity(set_images)
    return highest_intensity_image


def binarize(image):
    return np.where(image == image.max(), 1, 0)

def highlight_highest_intensity_region(original_image, segmented_image):
    # Find unique values in the segmented image
    unique_values = np.unique(segmented_image)

    # Initialize variables to keep track of the highest and lowest average intensities and their corresponding labels
    highest_avg_intensity = -1
    lowest_avg_intensity = float('inf')
    highest_intensity_label = None
    lowest_intensity_label = None

    # Iterate through unique values in the segmented image
    for label in unique_values:
        if label == 0:  # Skip background (if label 0 represents it)
            continue

        # Create a mask to select pixels with the current label
        mask = (segmented_image == label)

        # Calculate the average intensity of the original image for the current region
        average_intensity = np.mean(original_image[mask])

        # Check if the average intensity is higher or lower than the current highest or lowest
        if average_intensity > highest_avg_intensity:
            highest_avg_intensity = average_intensity
            highest_intensity_label = label
        if average_intensity < lowest_avg_intensity:
            lowest_avg_intensity = average_intensity
            lowest_intensity_label = label

    # Create a binary image where the region with the highest intensity is white, and others are black
    highest_intensity_region = (segmented_image == highest_intensity_label).astype(np.uint8)

    return highest_intensity_region



def test_process(image_path, t0: bool):
    # Load in the images 
    d_image = read_and_preprocess(image_path)
    # Histogram Equalize
    if (t0):
        d_image_eq = exposure.equalize_adapthist(d_image, clip_limit=0.03)
    else: 
        d_image_eq = histogram_eq(d_image)
    # Gabor Filter the image
    d_image_gab = gabor_filter(d_image_eq)
    # Ensure correct labeling 
    d_image_gab = highlight_highest_intensity_region(d_image_eq, d_image_gab)
    # Binarize
    d_image_gab = binarize(d_image_gab)
    # Filter out areas of void probable
    d_image_fil = np.where(d_image_gab == 1, d_image_eq, 0)

    # Median filter the images
    d_image_med = medfilt(d_image_fil)
    # Find the voiding
    if (t0):
        d_image_iso = void_isolation(d_image_med, False, 3, .001)
        print(np.unique(d_image_iso))
        return d_image_iso*127.5
    else: 
        d_image_iso = void_isolation(d_image_med, True, 3, .001)
        return d_image_iso

