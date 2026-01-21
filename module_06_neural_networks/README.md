# Module 06: Neural Networks (Layers)

## The Problem
"How do we find patterns inside patterns?"

## Simple Explanation
Imagine looking at a picture of a face.
1.  **Layer 1:** Sees lines and edges (Horizontal, Vertical).
2.  **Layer 2:** Connects lines to make shapes (Circles, Squares).
3.  **Layer 3:** Connects shapes to make parts (Eye, Nose).
4.  **Layer 4:** Recognizes the Face.

This code shows **one layer** doing the first step: Detecting edges.

## The Math Behind It (Class 12)
A "Convolution" is just the **Sum of Products** (Dot Product).
We slide a small $3 \times 3$ grid (Kernel $K$) over the image $I$:

$$ Output(x,y) = \sum \sum I(x+i, y+j) \times K(i,j) $$

*   If the image numbers match the kernel numbers, the sum is **BIG**.
*   If they are opposite, they cancel out to **ZERO**.
*   **ReLU Function:** $f(x) = \max(0, x)$. It just deletes negative numbers.

## Connection to LLMs
Deep Learning is just "many layers sandwich".
*   In Image AI (CNNs), layers look for visual shapes.
*   In Text AI (Transformers), layers look for *meaning* shapes (Grammar -> Logic -> Reasoning).
The math (multiplying numbers in a grid) is remarkably similar.

## What to Run
1.  `cnn.py`:
    *   **Input:** A simple image with a diagonal line.
    *   **Conv Output:** The computer highlights the edges.
    *   **ReLU Output:** It removes negative numbers (Cleaning up).
    *   **Pooling:** It shrinks the image (Summarizing).
