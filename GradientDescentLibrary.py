import numpy as np
from typing import Tuple

def computeMSE(xActual: np.array, yActual: np.array, beta0: float, beta1: float) -> float:
    """Calculate mean squared error."""
    predValues = (xActual * beta1) + beta0
    return np.mean((yActual - predValues) ** 2)

def nextGradientBetas(xActual: np.array, yActual: np.array, beta0: float, beta1: float) -> Tuple[float, float]:
    """Calculate gradients for beta0 and beta1."""
    predValues = (beta1 * xActual) + beta0
    gradBeta0 = -2 * np.mean(yActual - predValues)
    gradBeta1 = -2 * np.mean(xActual * (yActual - predValues))
    return gradBeta0, gradBeta1

def computeGradientDescent(xTrain: np.array, yTrain: np.array, xTest: np.array, yTest: np.array, eta: float, batchSize: int, maxIterations: int = 10000, tol: float = 1e-6) -> Tuple[float, float, np.array, np.array, float, np.array, int]:
    """Perform gradient descent to find optimal beta coefficients with different batch sizes."""
    steps = []
    bias = []
    variance = []
    
    # Initialize beta coefficients
    beta0, beta1 = np.random.normal(loc=0, scale=1, size=2)

    step = 0
    previousError = float('inf')

    # Iterating until convergence or maximum iterations
    for _ in range(maxIterations):
        for start in range(0, len(xTrain), batchSize):
            end = start + batchSize
            xBatch = xTrain[start:end]
            yBatch = yTrain[start:end]

            gradBeta0, gradBeta1 = nextGradientBetas(xBatch, yBatch, beta0, beta1)
            beta0 -= eta * gradBeta0
            beta1 -= eta * gradBeta1
        
        step += 1
        steps.append(step)
        bias.append(computeMSE(xTrain, yTrain, beta0, beta1))   
        variance.append(computeMSE(xTest, yTest, beta0, beta1)) 
        
        # Check for convergence
        currentError = bias[-1]
        if abs(previousError - currentError) < tol:
            print(f"Terminating: Change in error ({abs(previousError - currentError)}) is below tolerance ({tol}) for eta ({eta}) at step ({step})")
            break

        previousError = currentError
    
    return beta0, beta1, np.array(bias), np.array(variance), eta, np.array(steps), batchSize

def computeGradientDescentV1(xTrain: np.array, yTrain: np.array, xTest: np.array, yTest: np.array, eta: float, batchSize: int, maxIterations: int = 10000, tol: float = 1e-6) -> Tuple[float, float, np.array, np.array, float, np.array, int]:
    """Perform gradient descent to find optimal beta coefficients with different batch sizes."""
    steps = []
    bias = []
    variance = []
    
    # Initialize beta coefficients
    beta0, beta1 = np.random.normal(loc=0, scale=1, size=2)

    step = 0
    previousError = float('inf')

    # Iterating until convergence or maximum iterations
    for _ in range(maxIterations):
        for start in range(0, len(xTrain), batchSize):
            end = start + batchSize
            xBatch = xTrain[start:end]
            yBatch = yTrain[start:end]

            gradBeta0, gradBeta1 = nextGradientBetas(xBatch, yBatch, beta0, beta1)
            beta0 -= eta * gradBeta0
            beta1 -= eta * gradBeta1

            # Calculate bias and variance for the current batch
            bias.append(computeMSE(xTrain, yTrain, beta0, beta1))
            variance.append(computeMSE(xTest, yTest, beta0, beta1))

            # Update steps
            step += 1
            steps.append(step)

            # Check for convergence
            currentError = bias[-1]
            if abs(previousError - currentError) < tol:
                print(f"Terminating: Change in error ({abs(previousError - currentError)}) is below tolerance ({tol}) for eta ({eta}) at step ({step})")
                return beta0, beta1, np.array(bias), np.array(variance), eta, np.array(steps), batchSize

            previousError = currentError
    
    return beta0, beta1, np.array(bias), np.array(variance), eta, np.array(steps), batchSize
