import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, generate_linear_data, setup_plot

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def stochastic_gradient_descent_with_path(X, y, theta, learning_rate=0.01, iterations=50):
    cost_history = []
    theta_path = [theta.copy()]
    m = len(y)
    
    for i in range(iterations):
        for j in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            # Gradient for single example
            prediction = xi.dot(theta)
            error = prediction - yi
            gradient = xi.T.dot(error) # No 1/m factor for single example usually, or 1/1
            
            theta = theta - learning_rate * gradient
            theta_path.append(theta.copy())
            
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
    return theta, cost_history, np.array(theta_path)

if __name__ == "__main__":
    set_seed(42)
    print("Running Stochastic Gradient Descent...")
    
    # Generate data
    x_raw, y_raw = generate_linear_data(m=3, c=4, n=100, noise_factor=2)
    X = np.c_[np.ones((len(x_raw), 1)), x_raw]
    y = y_raw
    
    theta = np.random.randn(2)
    
    # Using very small LR because SGD is noisy and data scaling
    theta_final, cost_history, theta_path = stochastic_gradient_descent_with_path(X, y, theta, learning_rate=0.001, iterations=5)
    
    print(f"Final Theta: {theta_final}")
    
    # Plot Cost
    setup_plot("SGD Cost History (Noisy)", "Epoch", "Cost")
    plt.plot(cost_history)
    plt.show()
    
    # Visualize param path (first 200 updates)
    setup_plot("SGD Parameter Path (First 200 updates)", "Intercept", "Slope")
    plt.plot(theta_path[:200, 0], theta_path[:200, 1], 'r.-', alpha=0.5)
    plt.plot(theta_final[0], theta_final[1], 'k*', markersize=15, label="Final")
    plt.legend()
    plt.show()
