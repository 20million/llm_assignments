import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Callable, Dict, List, Tuple

# Function to generate target values with noise
def yGenerator(x: np.ndarray) -> Dict[float, float]:
    return {xi: (2 * (xi ** 4)) - (3 * (xi ** 3)) + (7 * (xi ** 2)) - (23 * xi) + 8 + np.random.normal(0, 3) for xi in x}

# Function to calculate beta values using linear regression
def deriveBeta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_transpose = X.T
    beta = np.linalg.inv(X_transpose @ X) @ (X_transpose @ y)
    return beta

# Function to predict Y values for a given degree model
def predictValues(betaValues: np.ndarray, xValues: np.ndarray) -> np.ndarray:
    degree = len(betaValues) - 1
    predictions = np.zeros_like(xValues)
    for i in range(degree + 1):
        predictions += betaValues[i] * (xValues ** i)
    return predictions

# Function to compute bias and variance
def computeMSE(predictions: np.ndarray, actualY: np.ndarray) -> Tuple[float, float]:
    mse = np.mean((predictions - actualY) ** 2)
    return mse

# Function to create feature matrices
def createFeatureMatrix(xValues: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([xValues ** i for i in range(degree + 1)])

# Function to plot function curves for all degrees
def plotFunctionCurves(trainX: np.ndarray, trainY: np.ndarray, xRange: np.ndarray, yActual: np.ndarray, betaValuesForDegrees: List[Tuple[int, np.ndarray]]) -> None:
    plt.figure(figsize=(10, 6))
    
    # # Plot actual function
    plt.plot(xRange, yActual, label='Actual Function', color='black', linewidth=2)
    
    # Plot training data points
    plt.scatter(trainX, trainY, label='Training Data', color='grey', alpha=0.6)
    
    # Plot polynomial approximations for each degree
    for degree, betaValues in betaValuesForDegrees:
        predictions = predictValues(betaValues, xRange)
        plt.plot(xRange, predictions, label=f'Degree {degree} Polynomial', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Curves for All Degrees')
    plt.legend()
    plt.show()

def plotBiasVariance(biasVarianceResults: List[Dict[str, float]]) -> None:
    plt.figure(figsize=(10, 6))

    # Extracting the degrees, training errors (bias), and testing errors (variance)
    degrees = [result['degree'] for result in biasVarianceResults]
    trainError = [result['trainBias'] for result in biasVarianceResults]  # Training error (bias)
    testError = [result['testVariance'] for result in biasVarianceResults]   # Testing error (variance)

    # Plotting training error (bias) across different degrees
    plt.plot(degrees, trainError, label='Training Error (Bias)', marker='o', color='blue')

    # Plotting testing error (variance) across different degrees
    plt.plot(degrees, testError, label='Testing Error (Variance)', marker='o', color='red')

    # Add labels and title
    plt.xlabel('Degree of Polynomial (Complexity)')
    plt.ylabel('Error')
    plt.title('Training and Testing Error across Different Degrees')

    # Add legend
    plt.legend()
    plt.show()

# Main function
def main() -> None:
    xRange = np.linspace(-5, 5, 100)
    # Generate dataset
    dataset = yGenerator(xRange)
    df = pd.DataFrame(dataset.items(), columns=['x', 'y'])

    # Split the dataset into training and test data
    trainDf, testDf = train_test_split(df, test_size=0.20, random_state=42)
    trainX = trainDf['x'].values
    trainY = trainDf['y'].values
    testX = testDf['x'].values
    testY = testDf['y'].values

    # Define degrees to analyze
    degrees = [1, 2, 3, 4]
    biasVarianceResults = []
    betaValuesForDegrees = []

    # Define the range for plotting
    yActual = (2 * (xRange ** 4)) - (3 * (xRange ** 3)) + (7 * (xRange ** 2)) - (23 * xRange) + 8

    # Calculate bias and variance for each degree and collect beta values
    for degree in degrees:
        # Create feature matrices for training data
        trainXMatrix = createFeatureMatrix(trainX, degree)

        # Calculate beta values using linear regression
        betaValues = deriveBeta(trainXMatrix, trainY)
        
        # Add beta values to the list
        betaValuesForDegrees.append((degree, betaValues))

        # Compute predictions for training and test data
        trainPredictions = predictValues(betaValues, trainX)
        testPredictions = predictValues(betaValues, testX)

        # Calculate bias and variance for training and test data
        trainBias = computeMSE(trainPredictions, trainY)
        testVariance = computeMSE(testPredictions, testY)

        # Append results to biasVarianceResults list
        biasVarianceResults.append({
            'degree': degree,
            'trainBias': trainBias,
            'testVariance': testVariance
        })

    # Plot function curves for all degrees
    plotFunctionCurves(trainX, trainY, xRange, yActual, betaValuesForDegrees)

    # Plot bias and variance across all degrees
    plotBiasVariance(biasVarianceResults)

    # Display bias and variance results for each polynomial degree
    for result in biasVarianceResults:
        print(f"Degree {result['degree']}:" +
              f"\n  Train Bias: {result['trainBias']}" +
              f"\n  Test Variance: {result['testVariance']}\n")

if __name__ == '__main__':
    main()
