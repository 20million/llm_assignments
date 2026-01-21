import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

"""
Concepts:
- Overfitting (High Variance) vs Underfitting (High Statistical Bias)
- Features (Polynomial expansion)
- Generalization

This file shows why we can't just "memorize" data.
- "Statistical Bias": The model is too simple (Straight line).
- "Variance": The model is too sensitive (Wiggly line).
- "Features": We create new inputs (x^2, x^3) from the original x to help the model learn curves.
LLMs need to find the balance: understanding patterns without memorizing the training data.
"""

def generate_polynomial_data(n=20, noise=1.0):
    """
    True function: y = 0.5 * x^2 + x + 2 + noise
    Using small range to keep it simple.
    """
    x = np.linspace(-3, 3, n)
    y_clean = 0.5 * x**2 + x + 2
    y_noise = y_clean + np.random.normal(0, noise, n)
    return x, y_noise, y_clean

def fit_polynomial(x, y, degree):
    """
    Fit polynomial of arbitrary degree using Gradient Descent (Batch).
    Hypothesis h(x) = w0 + w1*x + w2*x^2 + ...
    """
    # Create feature matrix
    # FEATURES: We are expanding the single input 'x' into multiple features [1, x, x^2, ... x^degree]
    # This allows a linear model (weighted sum) to fit curved data.
    # LLMs accept "tokens" as features and learn complex relationships between them.
    X_poly = np.column_stack([x**i for i in range(degree + 1)]) # includes bias x^0 (Intercept term)
    
    # Normal Equation (Closed Form) for stability in this demo instead of GD tuning
    # w = (X.T X)^-1 X.T y
    # Adding small regularization for stability with high degrees
    reg = 1e-5 * np.eye(degree + 1)
    reg[0,0] = 0
    w = np.linalg.inv(X_poly.T @ X_poly + reg) @ X_poly.T @ y
    return w

def predict_polynomial(x, w):
    degree = len(w) - 1
    X_poly = np.column_stack([x**i for i in range(degree + 1)])
    return X_poly @ w

if __name__ == "__main__":
    set_seed(42)
    print("Running Bias-Variance Tradeoff visualization...")
    
    x_train, y_train, _ = generate_polynomial_data(n=15, noise=2.0)
    x_test, y_test, _ = generate_polynomial_data(n=10, noise=2.0)
    
    x_plot = np.linspace(-3.2, 3.2, 100)
    
    degrees = [1, 2, 12]
    titles = ["Underfitting (High Bias)", "Just Right", "Overfitting (High Variance)"]
    
    plt.figure(figsize=(15, 5))
    
    for i, degree in enumerate(degrees):
        w = fit_polynomial(x_train, y_train, degree)
        
        y_train_pred = predict_polynomial(x_train, w)
        y_test_pred = predict_polynomial(x_test, w)
        y_plot_pred = predict_polynomial(x_plot, w)
        
        train_mse = np.mean((y_train - y_train_pred)**2)
        test_mse = np.mean((y_test - y_test_pred)**2)
        
        plt.subplot(1, 3, i+1)
        plt.scatter(x_train, y_train, color='blue', label='Train')
        plt.scatter(x_test, y_test, color='red', marker='x', label='Test')
        plt.plot(x_plot, y_plot_pred, color='green', label=f'Model deg={degree}')
        
        plt.title(f"{titles[i]}\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(-5, 15)
        plt.legend()
        
    plt.tight_layout()
    plt.show()
