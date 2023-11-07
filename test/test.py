import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import medfilt
from skimage import exposure
import os
from PIL import Image

import importlib.util

# Define the file path of the module you want to import
module1_path = 'app\\image_processing\\functions.py'  # Replace with the actual file path

# Load the module from the file path
spec = importlib.util.spec_from_file_location("module1", module1_path)
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)


def main():
    # Load in the images 
    d_image = pf.read_and_preprocess(r'C:\Users\antpoole\AMD\Void\test\test_data\csam_13.png')
    t0_image = pf.read_and_preprocess(r'C:\Users\antpoole\AMD\Void\test\test_data\Unit 64 Die 1.png')
    # Histogram Equalize
    d_image_eq = pf.histogram_eq(d_image)
    t0_image_eq = exposure.equalize_adapthist(t0_image, clip_limit=0.03)
    # Gabor Filter the image
    d_image_gab = pf.gabor_filter(d_image_eq)
    t0_image_gab = pf.gabor_filter(t0_image_eq)
    # Ensure that the void probable region is white and black is not void probable
    d_image_gab= pf.highlight_highest_intensity_region(d_image_eq, d_image_gab)
    t0_image_gab = pf.highlight_highest_intensity_region(t0_image_eq, t0_image_gab)
    # Binarize
    d_image_gab = pf.binarize(d_image_gab)
    t0_image_gab = pf.binarize(t0_image_gab)
    # Filter out areas of void probable
    d_image_fil = np.where(d_image_gab == 1, d_image_eq, 0)
    t0_image_fil = np.where(t0_image_gab == 1, t0_image_eq, 0)
    # Median filter the images
    d_image_med = medfilt(d_image_fil)
    t0_image_med =  medfilt(t0_image_fil)
    # Find the voiding
    d_image_iso = pf.void_isolation(d_image_med, True, 3, .001)
    t0_image_iso = pf.void_isolation(t0_image_med, False, 3, .001)
    plt.axis('off')
    plt.imshow(d_image_iso, cmap='gray')
    plt.show()
    plt.imshow(t0_image_iso, cmap='gray')
    plt.show()


def check_all(directory_path):
     # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # List all files in the directory
    file_list = os.listdir(directory_path)

    # Loop through the files and process images
    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)

        # Check if the file is an image based on its extension (you can customize this check)
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            try:
                # Open the image using Pillow (PIL)
                image = Image.open(file_path)

                # You can perform image processing or analysis here
                # For example, you can resize, filter, or analyze the image.
                processed_image = pf.test_process(file_path)
                plt.imshow(processed_image, cmap='gray')
                plt.show()
                # Close the image to release resources
                image.close()

                print(f"Processed: {file_name}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"Skipped non-image file: {file_name}")



# check_all(r'C:\Users\antpoole\AMD\Void\test\test_data')

main()
