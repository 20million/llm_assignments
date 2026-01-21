import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

def target_pdf(x):
    """
    Target distribution: Mixture of two Gaussians.
    0.3 * N(-2, 1) + 0.7 * N(2, 0.5)
    """
    return 0.3 * norm.pdf(x, loc=-2, scale=1) + 0.7 * norm.pdf(x, loc=2, scale=0.5)

def rejection_sampling(num_samples=1000, bounds=(-10, 10)):
    """
    Performs Rejection Sampling to sample from target_pdf.
    Proposal distribution: Uniform(bounds[0], bounds[1])
    M needs to be > max(target_pdf(x)) * (bounds[1]-bounds[0])?
    Wait, if q(x) is uniform, q(x) = 1/(b-a).
    We need M such that M*q(x) >= p(x) for all x.
    Max of p(x) around 2 is approx 0.7 * 1/(0.5*sqrt(2pi)) ~ 0.7 * 0.8 ~ 0.56.
    Let's pick M_prime = 1.0 (envelope height).
    """
    samples = []
    
    min_x, max_x = bounds
    
    # We simply pick a box [min_x, max_x] x [0, M_prime]
    # Proposal q(x) is uniform over [min_x, max_x].
    # We sample x ~ U(min_x, max_x)
    # We sample u ~ U(0, M_prime)
    # Accept if u < p(x).
    
    M_prime = 0.8 # Upper bound estimate for pdf
    
    count = 0
    while len(samples) < num_samples:
        x_candidate = np.random.uniform(min_x, max_x)
        u = np.random.uniform(0, M_prime)
        
        if u < target_pdf(x_candidate):
            samples.append(x_candidate)
        
        count += 1
        
    acceptance_rate = len(samples) / count
    print(f"Acceptance Rate: {acceptance_rate:.2%}")
    return np.array(samples)

if __name__ == "__main__":
    set_seed(42)
    print("Running Rejection Sampling...")
    
    samples = rejection_sampling(num_samples=2000, bounds=(-6, 6))
    
    x = np.linspace(-6, 6, 1000)
    true_pdf = target_pdf(x)
    
    setup_plot("Rejection Sampling: Mixture of Gaussians", "x", "Density")
    
    # Plot histogram of samples (normalized)
    plt.hist(samples, bins=50, density=True, alpha=0.5, label='Sampled Histogram', color='green')
    
    # Plot true PDF
    plt.plot(x, true_pdf, 'r-', linewidth=2, label='True PDF')
    
    plt.legend()
    plt.show()
