# LLM Assignments

This repository contains a collection of **Python scripts and Jupyter notebooks** exploring core machine-learning and statistical concepts that underpin modern **Generative AI and Large Language Models (LLMs)**.

Rather than a single cohesive application, this repo acts as a **learning sandbox**: each file focuses on a specific idea such as optimization, probability distributions, sampling, classification, or neural networks.

---

## Repository Structure

The repository is intentionally lightweight and mostly flat.

### Python Scripts

Each script is largely self-contained and demonstrates a core concept:

* **Optimization & Learning**

  * `GradientDescent.py`
  * `StochasticGradientDescent.py`
  * `GradientDescentByBatch.py`
  * `GradientDescentLibrary.py`

* **Classification & Models**

  * `BinaryClassifier.py`
  * `cnn.py` — example convolutional neural network experiment

* **Statistics & Probability**

  * `NormalDistribution.py`
  * `CDF.py`
  * `ClosedFormBeta.py`
  * `RandomSampler.py`

* **Analysis & Visualization**

  * `BiasVariancePlotter.py`
  * `EpsilonPlotter.py`

* **Algorithms & Utilities**

  * `StringDiffWithLCS.py` — string comparison using Longest Common Subsequence
  * `Utils.py` — shared helper functions

### Notebooks

* `jupyter/`
  Interactive Jupyter notebooks used to experiment with and visualize concepts implemented in the scripts.

### Dependencies

* `requirements.txt`
  Python dependencies required to run the scripts and notebooks.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/20million/llm_assignments.git
cd llm_assignments
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks

```bash
jupyter notebook
```

You can also run individual Python scripts directly:

```bash
python GradientDescent.py
```

---

## How to Use This Repo

This repository is best treated as a **concept library**, not a production codebase.

* Use the **gradient descent implementations** to understand optimization behavior.
* Use the **probability and sampling scripts** to build intuition for distributions and randomness.
* Use the **classifier and CNN examples** as stepping stones toward more complex models.
* Use the **visualization tools** to see how learning dynamics behave rather than just reading equations.

These fundamentals are essential groundwork before working with large neural networks and transformer-based models.

---

## Intended Audience

* Students learning machine learning or generative AI fundamentals
* Engineers refreshing core ML concepts
* Anyone who wants executable, minimal examples instead of heavy frameworks

---

## License

MIT License. See the `LICENSE` file for details.
