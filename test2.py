import cv2
import numpy as np
import os

image = cv2.imread('data/blurred/0066.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#Unsharp Masking: Original + (Original - Blurred) * Amount

amount = 1.5  # ~~ strength, adjustable
sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)

laplacian = cv2.Laplacian(sharpened, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

# Combine
enhanced = cv2.addWeighted(image, 1, cv2.merge([laplacian]*3), 0.7, 0)

cv2.imwrite('output/enhanced_sharp_0066.jpg', enhanced)
