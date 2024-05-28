import numpy as np
import matplotlib.pyplot as plt
# from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import load_img, img_to_array
from scipy.signal import convolve2d
from PIL import Image

# Load and preprocess the cropped image
croppedImagePath = 'rajeshk.jpg'
imgCropped = load_img(croppedImagePath, target_size=(100, 100))
imgArrayCropped = img_to_array(imgCropped) / 255.0

# Display the cropped image
plt.imshow(imgArrayCropped)
plt.title("Cropped Image")
plt.show()

# Load and preprocess the main image
imagePath = 'bigbangcrew.jpeg'
imgMain = load_img(imagePath, target_size=(2000, 2000))
imgArrayMain = img_to_array(imgMain) / 255.0

# Display the main image
plt.imshow(imgArrayMain)
plt.title("Main Image")
plt.show()

# Convert the images to grayscale for convolution using Pillow
imgCroppedGray = Image.fromarray((imgArrayCropped * 255).astype(np.uint8)).convert('L')
imgArrayCroppedGray = np.array(imgCroppedGray)/255

imgMainGray = Image.fromarray((imgArrayMain * 255).astype(np.uint8)).convert('L')
imgArrayMainGray = np.array(imgMainGray)/255

# Perform convolution using scipy.signal.convolve2d
convolutionResult = convolve2d(imgArrayMainGray, imgArrayCroppedGray, mode='same', boundary='wrap')

# Display the convolution result
plt.imshow(convolutionResult)
plt.title("Convolution Result")
plt.show()
