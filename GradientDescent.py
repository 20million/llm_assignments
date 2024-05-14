import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import NamedTuple, Tuple, List, Optional
from sklearn.model_selection import train_test_split

# Define a NamedTuple for a range of values
class SampleRange(NamedTuple):
    start: float
    stop: float
    count: int

# Define noise function
def generateNoise(mean: float, sigma: float) -> float:
    """Generates noise from a normal distribution with a given mean and standard deviation."""
    return np.random.normal(mean, sigma)

# Function to generate x and y values with noise
def generateData(xRange: SampleRange, noiseMean: float, noiseSigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generates x and y values within a given range with added noise."""
    xValues = np.linspace(xRange.start, xRange.stop, xRange.count)
    noise = generateNoise(noiseMean, noiseSigma)
    yValues = (2 * xValues) - 3 + noise
    return xValues, yValues

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

# Function to calculate the sum of squared errors
def calculateError(x: np.ndarray, y: np.ndarray, beta0: float, beta1: float) -> float:
    """Calculates the sum of squared errors."""
    predictions = beta0 + beta1 * x
    errors = y - predictions
    return np.mean(errors ** 2)

# Function to calculate the gradients for beta0 and beta1
def calculateGradients(x: np.ndarray, y: np.ndarray, beta0: float, beta1: float) -> Tuple[float, float]:
    """Calculates the gradients for beta0 and beta1."""
    predictions = beta0 + beta1 * x
    gradBeta0 = -2 * np.mean(y - predictions)
    gradBeta1 = -2 * np.mean(x * (y - predictions))
    return gradBeta0, gradBeta1

# Function to perform one iteration of gradient descent
def performGradientDescentIteration(x: np.ndarray, y: np.ndarray, beta0: float, beta1: float, learningRate: float) -> Tuple[float, float, float]:
    """Performs one iteration of gradient descent."""
    gradBeta0, gradBeta1 = calculateGradients(x, y, beta0, beta1)
    beta0 -= learningRate * gradBeta0
    beta1 -= learningRate * gradBeta1
    error = calculateError(x, y, beta0, beta1)
    return beta0, beta1, error

# Function to find beta values using gradient descent
def findBetaGradientDescent(x: np.ndarray, y: np.ndarray, learningRate: float, maxIterations: int = 10000, tol: float = 1e-6) -> Optional[Tuple[float, float, List[float]]]:
    """Finds beta values using the gradient descent optimization algorithm."""
    beta0, beta1 = np.random.normal(0, 1), np.random.normal(0, 1)
    previousError = calculateError(x, y, beta0, beta1)
    errors = [previousError]

    epochCount = 0

    for _ in range(maxIterations):
        beta0, beta1, error = performGradientDescentIteration(x, y, beta0, beta1, learningRate)
        error_difference = abs(errors[-1] - error)
        if error_difference < tol or error < 0.0001:
            if error_difference < tol:
                print(f"Terminating: Change in error ({error_difference:.6f}) is below tolerance ({tol:.6f}) at epoch=({epochCount}) and learningRate = ({learningRate}).")
            else:
                print(f"Terminating: Error ({error:.6f}) is below 0.0001 at epoch=({epochCount}) and learningRate = ({learningRate}).")
            break
        errors.append(error)
        epochCount+=1
        
    return beta0, beta1, errors

# Plot the data and gradient descent solution
def plotGradientDescentSolution(xValues: np.ndarray, yValues: np.ndarray, gradientDescentBetas: Tuple[float, float], learningRate: float) -> None:
    """Plots the original data points and the gradient descent solution."""
    plt.figure(figsize=(8, 6))
    plt.scatter(xValues, yValues, color='gray', label='Data', alpha=0.6)
    yGradientDescent = gradientDescentBetas[0] + gradientDescentBetas[1] * xValues
    plt.plot(xValues, yGradientDescent, color='blue', label='Gradient Descent Solution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"Gradient Descent Solution for learningRate ({learningRate})")
    plt.legend()
    plt.show()

# Plot the data, closed-form solution, and gradient descent solutions
def plotSolutions(xValues: np.ndarray, yValues: np.ndarray, closedFormBetas: np.ndarray, gradientDescentResults: List[Tuple[Tuple[float, float], List[float], float]], etas: List[float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(xValues, yValues, color='gray', label='Data', alpha=0.6)

    # Plot closed-form solution
    yClosedForm = closedFormBetas[0] + closedFormBetas[1] * xValues
    plt.plot(xValues, yClosedForm, color='red', label='Closed-Form Solution')

    # Plot gradient descent solutions
    for gradientDescentBetas, errors, learningRate in gradientDescentResults:
        yGradientDescent = gradientDescentBetas[0] + gradientDescentBetas[1] * xValues
        plt.plot(xValues, yGradientDescent, label=f'Gradient Descent (eta={learningRate})')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Comparison of Solutions')
    plt.legend()
    plt.show()

# Main function
def main():
    # Initialize the sample range and generate data
    xRange = SampleRange(start=-5, stop=5, count=100)
    xValues, yValues = generateData(xRange, noiseMean=0, noiseSigma=5)

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(xValues, yValues, test_size=0.2, random_state=42)

    # Calculate beta values using closed-form solution
    closedFormBetas = calculateClosedFormBeta(x_train, y_train, degree=1)
    print(f"closedFormBetas: ({closedFormBetas}).")

    # Define a list of different learning rates
    etas = [0.001, 0.01, 0.1]

    # Store results for each learning rate
    gradientDescentResults = []

    # Store training and testing errors for each learning rate
    training_errors = []
    testing_errors = []

    # Store all errors across solutions
    all_errors = []

    # Iterate over each learning rate
    for eta in etas:
        # Calculate beta values using gradient descent
        gradientDescentResult = findBetaGradientDescent(x_train, y_train, learningRate=eta)

        if gradientDescentResult:
            gradientDescentBetas, errors = gradientDescentResult[0:2], gradientDescentResult[2]
            # Store the results
            gradientDescentResults.append((gradientDescentBetas, errors, eta))
            all_errors.append(errors)
            
            # Calculate training error (bias)
            training_error = calculateError(x_train, y_train, gradientDescentBetas[0], gradientDescentBetas[1])
            training_errors.append(training_error)

            # Calculate testing error (variance)
            testing_error = calculateError(x_test, y_test, gradientDescentBetas[0], gradientDescentBetas[1])
            testing_errors.append(testing_error)

    # Plot all solutions together
    plotSolutions(xValues, yValues, closedFormBetas, gradientDescentResults, etas)

    # Plot all epoch/error across all solutions
    plt.figure(figsize=(10, 6))
    for errors, eta in zip(all_errors, etas):
        iterations = range(len(errors))
        plt.plot(iterations, errors, label=f'Learning Rate={eta}')

    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error over Epochs for All Solutions')
    plt.legend()
    plt.xlim(5, 250)
    plt.show()

    # Plot training and testing errors for all learning rates
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(training_errors)), training_errors, label='Training Error (Bias)')
    plt.plot(range(len(testing_errors)), testing_errors, label='Testing Error (Variance)')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training and Testing Errors for Different Learning Rates')
    plt.legend()
    plt.show()

 # Create DataFrame
    df_data = {
        "Method": ["Closed-Form"] + [f"Gradient Descent (eta={result[2]})" for result in gradientDescentResults],
        "Beta Values": [closedFormBetas] + [result[0] for result in gradientDescentResults],
        "Error": [0.0] + [result[1][-1] for result in gradientDescentResults],
        "Epochs": [np.nan] + [len(result[1]) for result in gradientDescentResults],
        "Learning Rate": [np.nan] + [result[2] for result in gradientDescentResults]
    }

    df = pd.DataFrame(df_data)
    print(df)

# Run the main function when the script is executed directly
if __name__ == "__main__":
    main()
