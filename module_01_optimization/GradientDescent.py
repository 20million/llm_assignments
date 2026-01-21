import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import from module_00_setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

"""
Concepts:
- Gradient Descent (Optimization)
- Loss Minimization
- Learning Rate

This file implements the core mechanism of "Learning".
We have a "Loss Function" (the curve), and we want to find the bottom.
The "x" value here represents a MODEL PARAMETER (Weight).
By updating "x" using the Gradient, we are "Training the Model".
This is exactly how LLMs tune their billions of parameters.
"""

def gradient_descent_1d(learning_rate=0.1, iterations=50):
    """
    Minimizes f(x) = x^2 using gradient descent.
    Derivative f'(x) = 2x.
    """
    x = 10.0 # Starting point
    history = [x]
    
    for _ in range(iterations):
        gradient = 2 * x
        x = x - learning_rate * gradient
        history.append(x)
        
    return np.array(history)

def plot_1d_optimization(history):
    setup_plot("1D Gradient Descent: f(x) = x^2", "Iteration", "x value")
    plt.plot(history, 'o-', label='x trajectory')
    plt.axhline(0, color='r', linestyle='--', label='Global Min (x=0)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    set_seed(42)
    print("Running 1D Gradient Descent...")
    history = gradient_descent_1d(learning_rate=0.1)
    
    print(f"Starting x: {history[0]}")
    print(f"Final x: {history[-1]}")
    
    plot_1d_optimization(history)
