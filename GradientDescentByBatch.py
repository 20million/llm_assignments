import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple, Tuple, List
from sklearn.model_selection import train_test_split
import pandas as pd
from Utils import SampleRange, generateNoise, generateData, createFeatureMatrixForErrorSurface
from GradientDescentLibrary import computeGradientDescent, computeMSE, computeGradientDescentV1

def computeBetaForClosedForm(xMatrix: np.array, yActual: np.array) -> np.array:
    """Calculates beta values using the closed-form solution for a polynomial of a given degree."""
    xTranspose = xMatrix.T
    beta = np.linalg.inv(xTranspose @ xMatrix) @ (xTranspose @ yActual)
    return beta

def plotEpochsErrorGraph(convergenceEpochs: List[int], biasList: List[np.array], varianceList: List[np.array], etaList: List[float], stepsList: List[np.array], batchSizes: List[int], alg: str) -> None:
    """Plot error rates during training."""
    plt.xlabel("Epochs (linear scale)")
    plt.ylabel("Error (linear scale)")
    plt.title(f"Error Rate During Training using ({alg}) Gradient Descent")
    
    for _, (bias, variance, eta, steps, batchSize) in enumerate(zip(biasList, varianceList, etaList, stepsList, batchSizes)):
        plt.plot(steps, bias, label=f"Bias at eta: {eta}, batch size: {batchSize}")
        plt.plot(steps, variance, label=f"Variance at eta: {eta}, batch size: {batchSize}")
    
    plt.legend()
    plt.figtext(0.5, 0.01, f'Visualize the convergence behavior of the ({alg}). Increase in the learning rate reduces the number of epochs.', fontsize=12, color='black', ha='center')
    plt.show()

def main():
    xRange = SampleRange(start=-5, stop=5, count=1000)
    xActual, yActual = generateData(xRange, generateNoise(mean=0, sigma=5, size=xRange.count))

    xTrain, xTest, yTrain, yTest = train_test_split(xActual, yActual, test_size=0.2)
    
    # Gradient descent with different batch sizes
    gdStochastic = computeGradientDescentV1(xTrain, yTrain, xTest, yTest, 0.001, batchSize=1)
    gdMiniBatch = computeGradientDescentV1(xTrain, yTrain, xTest, yTest, 0.001, batchSize=50)
    gdFullBatch = computeGradientDescentV1(xTrain, yTrain, xTest, yTest, 0.001, batchSize=len(xTrain))
    
    # Closed form solution
    xMatrix = createFeatureMatrixForErrorSurface(xTrain, degree=1)
    cf = computeBetaForClosedForm(xMatrix, yTrain)
    
    bias = computeMSE(xTrain, yTrain, cf[0], cf[1])
    variance = computeMSE(xTest, yTest, cf[0], cf[1])
    
    data = {
        "Method": ["Closed Form Solution", "Stochastic GD", "Mini-batch GD", "Full-batch GD"],
        "Bo": [cf[0], gdStochastic[0], gdMiniBatch[0], gdFullBatch[0]],
        "B1": [cf[1], gdStochastic[1], gdMiniBatch[1], gdFullBatch[1]],
        "Bias": [bias, gdStochastic[2][-1], gdMiniBatch[2][-1], gdFullBatch[2][-1]],
        "Variance": [variance, gdStochastic[3][-1], gdMiniBatch[3][-1], gdFullBatch[3][-1]],
        "Steps": ["-", gdStochastic[5][-1], gdMiniBatch[5][-1], gdFullBatch[5][-1]],
        "Eta": ["-", gdStochastic[4], gdMiniBatch[4], gdFullBatch[4]],
        "Epochs": ["-", gdStochastic[7], gdMiniBatch[7], gdFullBatch[7]]
    }
    
    df = pd.DataFrame(data)
    print(df)
    
    convergenceEpochs = [gdStochastic[5][-1], gdMiniBatch[5][-1], gdFullBatch[5][-1]]
    
    # Plot epochs error graph after calculations
    biasList = [gdStochastic[2]]
    varianceList = [gdStochastic[3]]
    etaList = [gdStochastic[4]]
    stepsList = [gdStochastic[5]]
    batchSizes = [gdStochastic[6]]
    
    plotEpochsErrorGraph(convergenceEpochs, biasList, varianceList, etaList, stepsList, batchSizes, 'stochastic')

    biasList = [gdMiniBatch[2]]
    varianceList = [gdMiniBatch[3]]
    etaList = [gdMiniBatch[4]]
    stepsList = [gdMiniBatch[5]]
    batchSizes = [gdMiniBatch[6]]
    
    plotEpochsErrorGraph(convergenceEpochs, biasList, varianceList, etaList, stepsList, batchSizes, 'minibatch')

    biasList = [gdFullBatch[2]]
    varianceList = [gdFullBatch[3]]
    etaList = [gdFullBatch[4]]
    stepsList = [gdFullBatch[5]]
    batchSizes = [gdFullBatch[6]]
    
    plotEpochsErrorGraph(convergenceEpochs, biasList, varianceList, etaList, stepsList, batchSizes, 'fullbatch')

if __name__ == "__main__":
    main()
