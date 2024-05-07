import numpy as np
import matplotlib.pyplot as plt

def plot_normal_distribution(mu, sigma, ax, label=None):
    """
    Plots the normal distribution for a given mean (mu) and standard deviation (sigma).
    
    Parameters:
        mu (float): The mean of the distribution.
        sigma (float): The standard deviation of the distribution.
        ax (matplotlib Axes): The Axes object to plot on.
        label (str): The label for the plot (used in the legend).
    """
    # Define x values for the range from mu - 5*sigma to mu + 5*sigma
    x_min = mu - 5 * sigma
    x_max = mu + 5 * sigma
    x = np.linspace(x_min, x_max, 1000)

    # Calculate the normal distribution
    normalization_factor = 1 / (sigma * np.sqrt(2 * np.pi))
    y = normalization_factor * np.exp(-((x - mu)**2) / (2 * sigma**2))

    # Plot the normal distribution on the given axes
    ax.plot(x, y, label=label)

# Initialize a figure and axes
fig, ax = plt.subplots()

# Plot different normal distributions
# Case 1: Different standard deviations, same mean
mu = 0
sigmas = [0.5, 1, 2]
for sigma in sigmas:
    plot_normal_distribution(mu, sigma, ax, label=f'μ = {mu}, σ = {sigma}')

# Case 2: Different means, same standard deviation
sigma = 1
means = [-2, 0, 2]
for mu in means:
    plot_normal_distribution(mu, sigma, ax, label=f'μ = {mu}, σ = {sigma}')

# Add title, labels, and legend
ax.set_title('Normal Distributions')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# Display the plot
plt.show()
