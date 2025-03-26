import cv2
import numpy as np
import os

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  # Return None if the image cannot be loaded
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

# Directory containing the images
directory = 'small-sample/'

# Loop through image numbers 301 to 334
for i in range(301, 335):
    image_path = os.path.join(directory, f"{i:04d}.jpg")  # Format to 4-digit filename
    sharpness_score = calculate_sharpness(image_path)
    if sharpness_score is not None:
        print(f"Sharpness Score for {i:04d}.jpg: {sharpness_score}")
    else:
        print(f"Could not read {i:04d}.jpg")
