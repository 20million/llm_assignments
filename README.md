# Foundations of Generative AI (From Scratch)

> **"If you want to bake a cake, don't buy a pre-mix. Buy flour and sugar."**

Welcome! If you are a student (Class 11/12 or College Freshman) and want to understand how **ChatGPT** or **Gemini** actually works, this course is for you.

We don't use magic buttons. We generally don't use heavy libraries like PyTorch here. We use **Math** and **Code** to build the engine parts one by one.

## What is this?

Imagine you want to understand how a car engine works.
*   **Approach A:** You drive a Ferrari. (This is using ChatGPT).
*   **Approach B:** You build a toy engine with batteries and magnets. (This is this course).

We will build the "toy engine" parts of an AI.

## Coding Requirement (Class 11/12 CS)

You only need to know:
*   Python Basics (Loops, Lists, Functions).
*   A little bit of Math (Coordinate Geometry, Probability).
*   **No prior AI knowledge needed.**

## The Journey

Follow these modules in order. Each one teaches you **one secret** about LLMs.

1.  **[Module 00: Setup](./module_00_setup/)** — **The Rules.** Why we need to be strict with computers.
2.  **[Module 01: Optimization](./module_01_optimization/)** — **Trial & Error.** How computers "learn" by making mistakes and fixing them. (Like practicing for Math boards).
3.  **[Module 03: Probability](./module_03_probability/)** — **Likelihood.** Why AI is never 100% sure, and why that makes it creative.
4.  **[Module 02: Bias & Variance](./module_02_bias_variance/)** — **Rote Learning vs Understanding.** Why memorizing the textbook (Overfitting) fails in the exam.
5.  **[Module 04: Classification](./module_04_classification/)** — **Yes/No.** How a computer decides if a photo is a "Cat" or "Not Cat".
6.  **[Module 05: Sequences](./module_05_sequences/)** — **Words & Order.** How we compare spellings and sentences.
7.  **[Module 06: Neural Networks](./module_06_neural_networks/)** — **The Brain.** How layers of filters help the computer "see" patterns.

## 📖 [Dictionary of Terms](./GLOSSARY.md)
Confused by "Weights", "Features", or "Bias"? **[Click here for the Glossary](./GLOSSARY.md)**.

## 📘 Concept Index: Where Core ML Terms Appear

*   **Linear Regression**
    *   Appears in: `module_01_optimization` (Gradient Descent), `module_02_bias_variance` (Polynomial Fitting)
    *   Implemented as: Minimizing squared error via gradient descent.

*   **Logistic Regression**
    *   Appears in: `module_04_classification/BinaryClassifier.py`
    *   Implemented as: Separating data with a sigmoid decision boundary.

*   **Features**
    *   Appears in: `module_02_bias_variance/BiasVariancePlotter.py`
    *   Implemented as: The input vectors ($x$, $x^2$, $x^3$...) used to train the model.

*   **Parameters / Weights**
    *   Appears in: `module_01_optimization`, `module_04_classification`
    *   Implemented as: The variables (theta, w) that the model updates during training.

*   **Bias (Offset Term)**
    *   Appears in: `module_04_classification/BinaryClassifier.py`, `module_06_neural_networks/cnn.py`
    *   Implemented as: The extra "intercept" term added to the weighted sum. (Distinguish from "Statistical Bias" below).

*   **Statistical Bias**
    *   Appears in: `module_02_bias_variance/BiasVariancePlotter.py`
    *   Implemented as: The error introduced by approximating a complex problem with a too-simple model.

*   **Activations**
    *   Appears in: `module_04_classification/BinaryClassifier.py` (Sigmoid), `module_06_neural_networks/cnn.py` (ReLU)
    *   Implemented as: Functions that squash or filter the output of a neuron.

*   **Network Architecture**
    *   Appears in: `module_06_neural_networks/cnn.py`
    *   Implemented as: The explicit arrangement of layers (Convolution -> ReLU -> Pooling).

*   **Sequences**
    *   Appears in: `module_05_sequences/StringDiffWithLCS.py`
    *   Implemented as: Comparing strings character-by-character using Dynamic Programming.

## How to Run

1.  Open your terminal/command prompt.
2.  Install the basic math tools:
    ```bash
    pip install -r module_00_setup/requirements.txt
    ```
3.  Run a file! Start with this:
    ```bash
    python module_01_optimization/GradientDescent.py
    ```

**Don't be afraid to break the code. Change numbers. See what happens.**
