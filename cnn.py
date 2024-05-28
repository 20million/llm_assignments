import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from scipy.signal import convolve2d
import cv2

# Load and preprocess the cropped image
croppedImagePath = 'howard.jpeg'
imgCropped = load_img(croppedImagePath, target_size=(100, 100))
imgArrayCropped = img_to_array(imgCropped) / 255.0

# Display the cropped image
plt.imshow(imgArrayCropped)
plt.title("Cropped Image")
plt.show()

# Load and preprocess the main image
imagePath = 'bigbangcrew.jpeg'
imgMain = load_img(imagePath, target_size=(100, 100))
imgArrayMain = img_to_array(imgMain) / 255.0

# Display the main image
plt.imshow(imgArrayMain)
plt.title("Main Image")
plt.show()

# Convert the images to grayscale for convolution
imgArrayCroppedGray = cv2.cvtColor((imgArrayCropped * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
imgArrayMainGray = cv2.cvtColor((imgArrayMain * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

# Perform convolution using scipy.signal.convolve2d
convolutionResult = convolve2d(imgArrayMainGray, imgArrayCroppedGray, mode='same', boundary='wrap')

# Display the convolution result
plt.imshow(convolutionResult, cmap='gray')
plt.title("Convolution Result")
plt.show()
