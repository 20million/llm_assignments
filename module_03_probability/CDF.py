import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

def gaussian_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def compute_cdf_numerically(x_values, mu, sigma):
    """
    Computes CDF via numerical integration (Riemann sum concept).
    """
    pdf_values = gaussian_pdf(x_values, mu, sigma)
    dx = x_values[1] - x_values[0] # Assume uniform spacing
    
    # Cumulative sum * width = integral from start
    # Note: This is an approximation starting from the first x point.
    cdf_values = np.cumsum(pdf_values) * dx
    return cdf_values

if __name__ == "__main__":
    set_seed(42)
    print("Running Numerical CDF visualization...")
    
    # Start from far left to capture 'all' probability mass
    x = np.linspace(-5, 5, 1000)
    mu = 0.0
    sigma = 1.0
    
    cdf = compute_cdf_numerically(x, mu, sigma)
    
    print(f"Final CDF value (should be approx 1.0): {cdf[-1]}")
    
    setup_plot(f"Normal Distribution CDF (Numerical)", "x", "Cumulative Probability")
    plt.plot(x, cdf, label='CDF')
    plt.axhline(1.0, color='r', linestyle='--', label='Max p=1.0')
    plt.axhline(0.0, color='r', linestyle='--')
    plt.grid(True)
    plt.legend()
    plt.show()
