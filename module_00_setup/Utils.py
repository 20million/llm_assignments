import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for reproducibility.
    
    Args:
        seed: The integer seed to allow for reproducible results.
    """
    np.random.seed(seed)
    print(f"Random seed set to {seed}")

def generate_linear_data(m: float, c: float, noise_factor: float = 1.0, n: int = 100):
    """
    Generates synthetic linear data: y = mx + c + noise.
    
    Args:
        m: Slope of the line.
        c: Y-intercept.
        noise_factor: Standard deviation of the Gaussian noise.
        n: Number of data points.
        
    Returns:
        tuple: (x, y) numpy arrays.
    """
    x = np.linspace(-10, 10, n)
    noise = np.random.normal(0, noise_factor, n)
    y = m * x + c + noise
    return x, y

def setup_plot(title: str, xlabel: str = "x", ylabel: str = "y"):
    """
    Configures a consistent plot style.
    
    Args:
        title: Title of the plot.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

if __name__ == "__main__":
    # Test the utils
    set_seed(42)
    x, y = generate_linear_data(m=2.0, c=5.0, noise_factor=2.0)
    
    setup_plot("Test Utils Plot")
    plt.scatter(x, y, label="Data", alpha=0.7)
    plt.plot(x, 2.0 * x + 5.0, color='red', linestyle='--', label="True Line")
    plt.legend()
    plt.show()
    print("Utils.py executed successfully.")
