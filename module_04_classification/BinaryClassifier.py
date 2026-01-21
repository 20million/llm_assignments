import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

"""
Concepts:
- Logistic Regression (Classification)
- Parameters (Weights & Bias)
- Activation Function (Sigmoid)
- Decision Boundary

This file implements a classifier.
- "Weights": Determine the angle of the line.
- "Bias (Offset)": Shifts the line from the origin.
- "Activation": Squashes the number into a probability (0 to 1).
LLMs use this exact "Dot Product + Bias -> Activation" logic in every single neuron.
"""

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def binary_cross_entropy_loss(y_true, y_pred):
    # Clip to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_logistic_regression(X, y, learning_rate=0.1, iterations=1000):
    m, n = X.shape
    weights = np.zeros(n)
    cost_history = []
    
    for i in range(iterations):
        # Forward
        # Z = Dot Product(Features, Weights) + (Bias is implicitly in X column 0)
        z = np.dot(X, weights)
        
        # ACTIVATION: Squash z into probability [0, 1]
        h = sigmoid(z)
        
        # Loss
        cost = binary_cross_entropy_loss(y, h)
        cost_history.append(cost)
        
        # Gradient
        # dJ/dw = (1/m) X.T (h - y)
        gradient = (1/m) * np.dot(X.T, (h - y))
        
        # Update
        weights -= learning_rate * gradient
        
    return weights, cost_history

if __name__ == "__main__":
    set_seed(42)
    print("Running Binary Classifier (Logistic Regression)...")
    
    # Generate 2D classification data
    n_samples = 200
    # Class 0: centered at (2, 2)
    x0 = np.random.randn(n_samples // 2, 2) + 2
    y0 = np.zeros(n_samples // 2)
    
    # Class 1: centered at (4, 4)
    x1 = np.random.randn(n_samples // 2, 2) + 4
    y1 = np.ones(n_samples // 2)
    
    X_raw = np.vstack((x0, x1))
    y = np.concatenate((y0, y1))
    
    y = np.concatenate((y0, y1))
    
    # Add bias (Offset term)
    # This '1' column corresponds to w0 (the Bias/Intercept parameter).
    # It allows the decision line to not pass through (0,0).
    X = np.c_[np.ones((n_samples, 1)), X_raw]
    
    # Train
    weights, cost_history = train_logistic_regression(X, y, learning_rate=0.1, iterations=1000)
    
    print(f"Final Weights: {weights}")
    
    # Plot Decision Boundary
    setup_plot("Logistic Regression Decision Boundary")
    
    # Scatter data
    plt.scatter(X_raw[y==0][:, 0], X_raw[y==0][:, 1], color='red', label='Class 0')
    plt.scatter(X_raw[y==1][:, 0], X_raw[y==1][:, 1], color='blue', label='Class 1')
    
    # Boundary: w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1) / w2
    x1_min, x1_max = X_raw[:, 0].min() - 1, X_raw[:, 0].max() + 1
    x1_vals = np.linspace(x1_min, x1_max, 100)
    x2_vals = -(weights[0] + weights[1] * x1_vals) / weights[2]
    
    plt.plot(x1_vals, x2_vals, 'k--', label='Decision Boundary')
    plt.legend()
    plt.show()
    
    # Plot Loss
    setup_plot("Training Loss", "Iteration", "BCE Loss")
    plt.plot(cost_history)
    plt.show()
