import cv2

image = cv2.imread('data/blurred/0066.jpg')

# Apply Laplacian on each color channel
b, g, r = cv2.split(image)  # Split into Blue, Green, and Red channels
laplacian_b = cv2.Laplacian(b, cv2.CV_64F)
laplacian_g = cv2.Laplacian(g, cv2.CV_64F)
laplacian_r = cv2.Laplacian(r, cv2.CV_64F)

# Convert back to uint8
laplacian_b = cv2.convertScaleAbs(laplacian_b)
laplacian_g = cv2.convertScaleAbs(laplacian_g)
laplacian_r = cv2.convertScaleAbs(laplacian_r)

laplacian_merged = cv2.merge((laplacian_b, laplacian_g, laplacian_r))
enhanced_image = cv2.addWeighted(image, 1, laplacian_merged, 1, 0)


cv2.imwrite('output/colored_laplacian_0066.jpg', laplacian_merged)
cv2.imwrite('output/enhanced_0066.jpg', enhanced_image)

# cv2.imshow('Original', image)
# cv2.imshow('Laplacian Colored', laplacian_merged)
# cv2.imshow('Enhanced Image', enhanced_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
