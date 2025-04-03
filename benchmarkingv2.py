import cv2
import numpy as np
import os

def calculate_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  # Return None if the image cannot be loaded
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return laplacian.var()

def calculate_tenengrad(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return (sobel_x**2 + sobel_y**2).mean()

def calculate_brenner(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    shifted = np.roll(image, -2, axis=1)
    return ((image - shifted)[:-2] ** 2).sum()

def calculate_fft_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    return np.sum(magnitude)

# Directory containing the images
directory = 'images/'
outfile = 'sharpness_results.txt'

# Open file to save results
with open(outfile, 'w') as f:
    f.write("Image Name\tLaplacian\tTenengrad\tBrenner\tFFT\n")
    
    for i in range(1, 334):
        image_name = f"{i:04d}.jpg"
        image_path = os.path.join(directory, image_name)
        
        laplacian_score = calculate_sharpness(image_path)
        tenengrad_score = calculate_tenengrad(image_path)
        brenner_score = calculate_brenner(image_path)
        fft_score = calculate_fft_sharpness(image_path)
        
        if None not in (laplacian_score, tenengrad_score, brenner_score, fft_score):
            f.write(f"{image_name}\t{laplacian_score}\t{tenengrad_score}\t{brenner_score}\t{fft_score}\n")
            print(f"{image_name}: Laplacian={laplacian_score}, Tenengrad={tenengrad_score}, Brenner={brenner_score}, FFT={fft_score}")
        else:
            print(f"Could not read {image_name}")
