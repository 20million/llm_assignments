import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple, Tuple, List
from sklearn.model_selection import train_test_split
import pandas as pd

class SampleRange(NamedTuple):
    start: float
    stop: float
    count: int

def generateNoise(mean: float, sigma: float) -> float:
    """Generate random noise."""
    return np.random.normal(mean, sigma)

def generateData(xRange: SampleRange, noiseMean: float, noiseSigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data."""
    xValues = np.linspace(xRange.start, xRange.stop, xRange.count)
    noise = generateNoise(noiseMean, noiseSigma)
    yValues = (2 * xValues) - 3 + noise
    return xValues, yValues

def findBetaClosedForm(xMatrix: np.array, yActual: np.array) -> np.array:
    """Find beta coefficients using closed-form solution."""
    xMatrix = np.array(xMatrix)
    xTranspose = xMatrix.transpose()
    firstPart = np.linalg.inv(np.matmul(xTranspose, xMatrix))
    secondPart = np.matmul(xTranspose, yActual)
    return np.matmul(firstPart, secondPart)

def findError(xActual: np.array, yActual: np.array, beta0: float, beta1: float) -> float:
    """Calculate mean squared error."""
    predValues = (xActual * beta1) + beta0
    return np.mean((yActual - predValues) ** 2)

def findBetas(xActual: np.array, yActual: np.array, beta0: float, beta1: float) -> float:
    """Calculate gradients for beta0 and beta1."""
    predValues = (beta1 * xActual) + beta0
    gradBeta0 = -2 * np.mean(yActual - predValues)
    gradBeta1 = -2 * np.mean(xActual * (yActual - predValues))
    return gradBeta0, gradBeta1

def findBetaGradientDescent(xTrain: np.array, yTrain: np.array, xTest: np.array, yTest: np.array, eta: float, maxIterations: int = 10000, tol: float = 1e-6) -> Tuple[float, float, np.array, np.array, np.array]:
    """Perform stochastic gradient descent to find optimal beta coefficients."""
    steps = []
    bias = []
    variance = []
    
    # Initialize beta coefficients
    beta0, beta1 = np.random.normal(loc=0, scale=1, size=2)
    randomIndices = np.random.choice(len(xTrain), size=len(xTrain), replace=True)

    step = 0
    previous_error = float('inf')

    # Iterating until convergence or maximum iterations
    for i in randomIndices:
        betas = findBetas(xTrain[i], yTrain[i], beta0, beta1)
        beta0 -= eta * betas[0]
        beta1 -= eta * betas[1]
        step += 1
        steps.append(step)
        bias.append(findError(xTrain, yTrain, beta0, beta1))   
        variance.append(findError(xTest, yTest, beta0, beta1)) 
        
        # Check for convergence
        if step >= maxIterations:
            print(f"Terminating: Maximum iterations reached ({maxIterations}).")
            break
        
        current_error = bias[-1]
        if abs(previous_error - current_error) < tol:
            print(f"Terminating: Change in error ({abs(previous_error - current_error)}) is below tolerance ({tol}) for eta ({eta}) at step ({step})")
            break

        previous_error = current_error

    # skip 5x, 5y
    plt.plot(steps[5:], bias[5:], label=f"Bias at eta: {eta}")
    plt.plot(steps[5:], variance[5:], label=f"Variance at eta: {eta}")
    
    return beta0, beta1, np.array(bias), np.array(variance), eta, np.array(steps)

def plotEpochsErrorGraph(convergenceEpochs: Tuple[int]) -> None:
    """Plot error rates during training."""
    plt.xlabel("Epochs (linear scale)")
    plt.ylabel("Epsilon (linear scale)")
    plt.title("Error Rate During Training using Stochastic Gradient Descent")
    plt.legend()
    plt.figtext(0.5, 0.01, 'Visualise the convergence behavior of the SGD algorithm. Increase in the learning rate reduces the number of epochs.', fontsize=12, color='black', ha='center')
    
    # # Add marker for convergence for each run
    # for epoch in convergenceEpochs:
    #     plt.axvline(x=epoch, color='r', linestyle='--', label='Convergence')
       
    plt.show()

def main():
    xRange = SampleRange(start=-5, stop=5, count=1000)
    xActual, yActual = generateData(xRange, noiseMean=0, noiseSigma=5)

    xTrain, xTest, yTrain, yTest = train_test_split(xActual, yActual, test_size=0.2)
    
    gd = findBetaGradientDescent(xTrain, yTrain, xTest, yTest, 0.001)
    gd1 = findBetaGradientDescent(xTrain, yTrain, xTest, yTest, 0.01)
    
    xMatrix = [[1, i] for i in xTrain]
    cf = findBetaClosedForm(xMatrix, yTrain)
    
    bias = findError(xTrain, yTrain, cf[0], cf[1])
    variance = findError(xTest, yTest, cf[0], cf[1])
    
    data = {
        "Method": ["Closed Form Solution", "SGD ETA .001", "SGD ETA .01"],
        "Bo": [cf[0], gd[0], gd1[0]],
        "B1": [cf[1], gd[1], gd1[1]],
        "Bias": [bias, gd[2][-1], gd1[2][-1]],
        "Variance": [variance, gd[-3][-1], gd1[-3][-1]],
        "Steps": ["-", gd[-1][-1], gd1[-1][-1]],
        "Eta": ["-", gd[-2], gd1[-2]]
    }
    
    df = pd.DataFrame(data)
    print(df)
    convergenceEpochs = [gd[-1][-1], gd1[-1][-1]]
    plotEpochsErrorGraph(convergenceEpochs)

if __name__ == "__main__":
    main()
