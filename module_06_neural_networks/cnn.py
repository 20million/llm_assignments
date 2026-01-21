import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

"""
Concepts:
- Neural Network Architecture (Layers)
- Convolution (Feature Extraction)
- Activation (ReLU - Non-linearity)
- Pooling (Summarization)

This file demonstrates a single "Forward Pass" of a deep network layer.
- "Convolution": Finds patterns (features) like edges.
- "ReLU": Removes noise (negative values). This is an ACTIVATION function.
- "Pooling": Summarizes the result.
Deep Learning is just stacking these blocks on top of each other.
"""

def convolve2d(image, kernel, stride=1, padding=0):
    """
    Performs 2D convolution via sliding window.
    """
    # Add padding
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape
    
    out_h = (i_h - k_h) // stride + 1
    out_w = (i_w - k_w) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for y in range(0, out_h):
        for x in range(0, out_w):
            start_y = y * stride
            start_x = x * stride
            # Extract region
            region = image[start_y : start_y + k_h, start_x : start_x + k_w]
            # Element-wise multiply and sum
            output[y, x] = np.sum(region * kernel)
            
    return output

def relu(x):
    # ACTIVATION: Rectified Linear Unit.
    # It introduces "non-linearity", allowing the network to learn complex shapes.
    # Without this, the network would just be one big Linear Regression.
    return np.maximum(0, x)

def max_pooling(feature_map, size=2, stride=2):
    """
    Performs Max Pooling.
    """
    h, w = feature_map.shape
    out_h = (h - size) // stride + 1
    out_w = (w - size) // stride + 1
    
    output = np.zeros((out_h, out_w))
    
    for y in range(out_h):
        for x in range(out_w):
            start_y = y * stride
            start_x = x * stride
            region = feature_map[start_y : start_y + size, start_x : start_x + size]
            output[y, x] = np.max(region)
            
    return output

if __name__ == "__main__":
    set_seed(42)
    print("Running CNN Manual Forward Pass...")
    
    # 1. Create a dummy image (5x5) - e.g., a diagonal line
    image = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1]
    ])
    
    # 2. Define a Kernel (Edge detector)
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    
    print("Original Image:")
    print(image)
    print("\nKernel:")
    print(kernel)
    
    # 3. Convolution
    conv_out = convolve2d(image, kernel, padding=1)
    print("\nConvolution Output:")
    print(conv_out)
    
    # 4. ReLU
    relu_out = relu(conv_out)
    print("\nReLU Output:")
    print(relu_out)
    
    # 5. Max Pooling
    pool_out = max_pooling(relu_out, size=2, stride=1)
    print("\nMax Pooling Output (stride=1):")
    print(pool_out)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Input")
    
    axes[1].imshow(conv_out, cmap='gray')
    axes[1].set_title("Conv (Edge Detect)")
    
    axes[2].imshow(relu_out, cmap='gray')
    axes[2].set_title("ReLU")
    
    axes[3].imshow(pool_out, cmap='gray')
    axes[3].set_title("Max Pool")
    
    plt.tight_layout()
    plt.show()
