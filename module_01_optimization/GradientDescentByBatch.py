import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, generate_linear_data, setup_plot

def mse_loss_grad(X, y, theta):
    """
    Computes Gradient of MSE w.r.t theta.
    J = (1/2m) * sum((h(x) - y)^2)
    Grad = (1/m) * X.T * (X*theta - y)
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = (1/m) * X.T.dot(errors)
    return gradient

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def mini_batch_gradient_descent(X, y, theta, learning_rate=0.01, iterations=50, batch_size=32):
    cost_history = []
    m = len(y)
    
    for i in range(iterations):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]
            
            gradient = mse_loss_grad(X_batch, y_batch, theta)
            theta = theta - learning_rate * gradient
            
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
    return theta, cost_history

if __name__ == "__main__":
    set_seed(42)
    print("Running Mini-Batch Gradient Descent...")
    
    # Generate data
    x_raw, y_raw = generate_linear_data(m=3, c=4, n=200, noise_factor=2)
    
    # Preprocess
    X = np.c_[np.ones((len(x_raw), 1)), x_raw] # Add bias term
    y = y_raw
    
    # Init theta
    theta = np.random.randn(2)
    
    # Run Mini-Batch GD
    # Small learning rate because data is wide (-10 to 10)
    theta_final, cost_history = mini_batch_gradient_descent(X, y, theta, learning_rate=0.01, iterations=100, batch_size=16)
    
    print(f"Final Theta: {theta_final}")
    print(f"True Params: Intercept=4, Slope=3")
    
    # Plot Cost
    setup_plot("Mini-Batch GD Cost History", "Iteration", "Cost (MSE)")
    plt.plot(cost_history)
    plt.show() # Blocks
