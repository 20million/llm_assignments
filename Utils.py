import numpy as np
from typing import NamedTuple, Tuple


class SampleRange(NamedTuple):
    start: float
    stop: float
    count: int

def generateNoise(mean: float, sigma: float) -> float:
    """Generate random noise."""
    return np.random.normal(mean, sigma)

def generateData(xRange: SampleRange, noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data."""
    xValues = np.linspace(xRange.start, xRange.stop, xRange.count)
    yValues = (2 * xValues) - 3 + noise
    return xValues, yValues

# Calculate x/feature matrix for the closed-form solution
def createFeatureMatrixForErrorSurface(x: np.ndarray, degree: int) -> np.ndarray:
    """Creates the x matrix for the closed-form solution."""
    return np.array([[xi ** j for j in range(degree + 1)] for xi in x])
