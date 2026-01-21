import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

if __name__ == "__main__":
    set_seed(99)
    print("Running Irreducible Error (Epsilon) visualization...")
    
    # Scenario: True function is f(x). Observed y = f(x) + epsilon.
    # We can model f(x) perfectly, but we can never eliminate epsilon.
    
    x = np.linspace(0, 10, 50)
    true_f = lambda x: 3 * np.sin(x)
    
    # Epsilon (noise)
    epsilon = np.random.normal(0, 1.5, size=len(x))
    
    y_observed = true_f(x) + epsilon
    
    setup_plot("Irreducible Error (Epsilon)", "x", "y")
    
    plt.plot(x, true_f(x), 'k--', linewidth=2, label='True Function f(x)')
    plt.scatter(x, y_observed, color='blue', alpha=0.6, label='Observed Data y = f(x) + ε')
    
    # Visualize the error bars (epsilon)
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [true_f(x[i]), y_observed[i]], 'r-', alpha=0.3)
        
    plt.plot([], [], 'r-', label='Residuals (Noise/Epsilon)')
    
    plt.legend()
    plt.show()
