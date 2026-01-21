# Module 04: Classification (Yes or No?)

## The Problem
"Is this email Spam or Not?"

## Simple Explanation
This is the most basic decision a computer makes.
Imagine a graph with red dots (Spam) and blue dots (Not Spam).
We want to draw a **line** separating them.
*   If a new email lands on the "Red" side -> Spam Folder.
*   If it lands on the "Blue" side -> Inbox.

## The Math Behind It (Class 12)
We take a linear combination of inputs ($z = mx + c$) and squash it using the **Sigmoid Function** $\sigma(z)$:
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

*   If $z$ is a large positive number, $\sigma(z) \approx 1$ (100% Yes).
*   If $z$ is a large negative number, $\sigma(z) \approx 0$ (0% No).
*   If $z = 0$, $\sigma(z) = 0.5$ (Unsure).

The "Line" is where $\sigma(z) = 0.5$.

## Connection to LLMs
At the very, very end of ChatGPT, there is a "Classifier".
It looks at everything you typed and asks:
"Is the next word 'Apple'? Yes/No. Is it 'Banana'? Yes/No."
It does this for every word in the dictionary!

## What to Run
1.  `BinaryClassifier.py`:
    *   Watch the loss go down.
    *   See the **Decision Boundary** (the dotted line) separate the red dots from the blue dots.
    *   That line is where the computer goes from "Probably No" to "Probably Yes".
