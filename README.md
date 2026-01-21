# Foundations of Generative AI (From Scratch)

This repository contains a "no-nonsense" foundations course on Generative AI. The core philosophy is **mechanical sympathy**: implementing optimization, probability, and neural networks from scratch (using only Numpy) to build deep intuition before using high-level frameworks.

## Course Philosophy

*   **No Magic:** We do not use PyTorch, TensorFlow, or Scikit-learn. Everything is implemented explicitly.
*   **Visual Intuition:** Every concept is accompanied by a visualization to verify understanding.
*   **One Concept, One File:** Code is modular and focused.

## Directory Structure

### Module 00: Setup
*   `Utils.py`: Helper functions for deterministic seeding and plotting.
*   `requirements.txt`: Minimal dependencies (`numpy`, `matplotlib`, `scipy`).

### Module 01: Optimization
*   `GradientDescent.py`: Basic scalar/vector gradient descent visualization.
*   `GradientDescentByBatch.py`: Mini-batch gradient descent loop.
*   `StochasticGradientDescent.py`: SGD visualization showing noisy updates.
*   `GradientDescentLibrary.py`: A reusable Class encapsulating GD logic.

### Module 02: Bias & Variance
*   `BiasVariancePlotter.py`: Polynomial regression showing underfitting vs overfitting.
*   `EpsilonPlotter.py`: Visualization of irreducible error.

### Module 03: Probability
*   `NormalDistribution.py`: Gaussian PDF implemented from scratch.
*   `CDF.py`: Numerical integration of PDF to compute CDF.
*   `ClosedFormBeta.py`: Beta distribution PDF using Gamma function.
*   `RandomSampler.py`: Rejection sampling to generate samples from arbitrary distributions.

### Module 04: Classification
*   `BinaryClassifier.py`: Logistic regression with explicit cross-entropy loss and gradient. Classification boundary visualization.

### Module 05: Sequences
*   `StringDiffWithLCS.py`: Longest Common Subsequence using Dynamic Programming (table visualization).

### Module 06: Neural Networks
*   `cnn.py`: Numpy-only 2D Convolution, ReLU, and Max Pooling forward pass.

## Usage

Each script is standalone. Run them from the root directory:

```bash
python module_01_optimization/GradientDescent.py
python module_06_neural_networks/cnn.py
```

## Dependencies

```bash
pip install -r module_00_setup/requirements.txt
```
