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
    plt.scatter(xValues, yValues, color='gray', label='Data')

    # Plot closed-form solution
    m = closedFormBetas[1]
    c = closedFormBetas[0]
    yClosedForm = m * xValues + c
    plt.plot(xValues, yClosedForm, color='red', label='Closed-Form Solution')
   
    # Plot boundary
    yPerpendicular = (-1/m) * xValues + c
    print(f'yPerpendicular={yPerpendicular}')
    plt.plot(xValues, yPerpendicular, color='black', label='boundary', linestyle='--')
    plt.scatter(xValues, binaryClassifier, color='blue', label='Binary Predictions')
    plt.ylim(-2,2)
    
    plt.axvline(x=0, color='purple')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('closed form')
    plt.legend()
    plt.show()

# Convert predicted values to binary predictions using a threshold
def predictBinary(y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_pred > threshold).astype(int)

# Function to calculate MCE for a given threshold
def calculate_mce(y_pred: np.ndarray, y_true: np.ndarray, threshold: float) -> float:
    binary_predictions = (y_pred > threshold).astype(int)
    incorrect_predictions = np.abs(binary_predictions - y_true)
    # mce = np.mean(incorrect_predictions)
    mce = np.sum(incorrect_predictions)
    return mce

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

    # Calculate MCE for different thresholds
    thresholds = np.arange(-2, 7)
    mce_values = [calculate_mce(yCap, yValues, threshold) for threshold in thresholds]
    print(f"thresholds: ({thresholds}).")
    print(f"mce_values: ({mce_values}).")

    # Plot MCE vs. Threshold
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, mce_values, color='blue')
    plt.xlabel('Threshold')
    plt.ylabel('Mean Classification Error (MCE)')
    plt.title('MCE vs. Threshold')
    plt.grid(True)  
    plt.show()


# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
