import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import NamedTuple, Tuple, List, Optional
from sklearn.model_selection import train_test_split

# Calculate x/feature matrix for the closed-form solution
def createFeatureMatrixForErrorSurface(x: np.ndarray, degree: int) -> np.ndarray:
    """Creates the x matrix for the closed-form solution."""
    return np.array([[xi ** j for j in range(degree + 1)] for xi in x])

# Function to calculate beta values using closed-form solution
def calculateClosedFormBeta(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Calculates beta values using the closed-form solution for a polynomial of a given degree."""
    xMatrix = createFeatureMatrixForErrorSurface(x, degree)
    xTranspose = xMatrix.T
    beta = np.linalg.inv(xTranspose @ xMatrix) @ (xTranspose @ y)
    return beta

# Plot the data, closed-form solution, and gradient descent solutions
def plotSolutions(xValues: np.ndarray, yValues: np.ndarray, closedFormBetas: np.ndarray, binaryClassifier: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(xValues, yValues, color='gray', label='Data', alpha=0.6)

    # Plot closed-form solution
    yClosedForm = closedFormBetas[0] + closedFormBetas[1] * xValues
    plt.plot(xValues, yClosedForm, color='red', label='Closed-Form Solution')
    plt.scatter(xValues, binaryClassifier, color='blue', label='Binary Predictions', alpha=0.6)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('closed form')
    plt.legend()
    plt.show()

# Convert predicted values to binary predictions using a threshold
def predictBinary(y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_pred > threshold).astype(int)

# Main function
def main():
    xValues = np.array([1,2,3,4,5, -3, -4, -5, -6])
    yValues = np.array([1,1,1,1,1,0,0,0,0])
    # Calculate beta values using closed-form solution
    closedFormBetas = calculateClosedFormBeta(xValues, yValues, degree=1)
    yCap = closedFormBetas[0] + closedFormBetas[1] * xValues
    print(f"closedFormBetas: ({closedFormBetas}).")
    print(f"yCap: ({yCap}).")
    print(f"yValues.mean(): ({yValues.mean()}).")
    bClass = predictBinary(yCap, yValues.mean())
    print(f"binaryClassifier: ({bClass}).")

    plotSolutions(xValues, yValues, closedFormBetas, bClass )

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
