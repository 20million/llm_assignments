import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

def beta_pdf(x, alpha_param, beta_param):
    """
    Computes Beta PDF.
    f(x) = (x^(alpha-1) * (1-x)^(beta-1)) / B(alpha, beta)
    where B(alpha, beta) = gamma(alpha)*gamma(beta) / gamma(alpha+beta)
    Defined for x in [0, 1].
    """
    if np.any((x < 0) | (x > 1)):
        # Just return 0 for out of bounds (simplified) or handle
        pass 
    
    # Beta function B(alpha, beta)
    B = (gamma(alpha_param) * gamma(beta_param)) / gamma(alpha_param + beta_param)
    
    return (x**(alpha_param - 1) * (1 - x)**(beta_param - 1)) / B

if __name__ == "__main__":
    set_seed(42)
    print("Running Closed-Form Beta Distribution visualization...")
    
    x = np.linspace(0.001, 0.999, 1000) # Avoid 0/1 singularities for plotting
    
    params = [
        (0.5, 0.5),
        (2.0, 2.0),
        (2.0, 5.0),
        (5.0, 1.0)
    ]
    
    setup_plot("Beta Distributions", "x", "Density")
    
    for alpha, beta in params:
        y = beta_pdf(x, alpha, beta)
        plt.plot(x, y, label=f"alpha={alpha}, beta={beta}")
        
    plt.legend()
    plt.show()
