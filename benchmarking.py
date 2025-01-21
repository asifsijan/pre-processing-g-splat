import cv2
import numpy as np

def calculate_sharpness(image_path):
    # Read in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Compute the Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # Calculate variance
    sharpness = laplacian.var()
    return sharpness

#very blurry- 

sharpness_score = calculate_sharpness('data/blurred/0066.jpg')
print(f"Sharpness Score og: {sharpness_score}")

#blurry one after some work

sharpness_score = calculate_sharpness('data/output/enhanced_sharp_0066.jpg')
print(f"Sharpness Score mod: {sharpness_score}")

#normal one

sharpness_score = calculate_sharpness('data/normal/0039.jpg')
print(f"Sharpness Score normal: {sharpness_score}")


#test result: 
# Sharpness Score og: 22.099880032684446
# Sharpness Score mod: 192.9992032472531
# Sharpness Score normal: 48.340878644952745
