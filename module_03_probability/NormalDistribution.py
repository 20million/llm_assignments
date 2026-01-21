import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

def gaussian_pdf(x, mu, sigma):
    """
    Computes the Probability Density Function (PDF) of a Gaussian.
    f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu)/sigma)^2)
    """
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma)**2
    return coefficient * np.exp(exponent)

if __name__ == "__main__":
    set_seed(42)
    print("Running Normal Distribution PDF visualization...")
    
    x = np.linspace(-5, 5, 1000)
    mu = 0.0
    sigma = 1.0
    
    pdf = gaussian_pdf(x, mu, sigma)
    
    setup_plot(f"Normal Distribution PDF (mu={mu}, sigma={sigma})", "x", "Density")
    plt.plot(x, pdf, label='PDF')
    plt.fill_between(x, pdf, alpha=0.2)
    plt.legend()
    plt.show()
