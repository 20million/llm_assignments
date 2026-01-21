# Module 02: Bias & Variance (Memorizing vs Understanding)

## The Problem
"Should I memorize the Guide Book or understand the Concept?"

## Simple Explanation
*   **High Bias (Underfitting):** You didn't study. You just guess "C" for every answer. You fail.
*   **High Variance (Overfitting):** You memorized the *exact* questions in the previous year's paper. But in the final exam, the numbers changed! You panic and fail.
*   **Good Fit:** You understood the formula. You can solve *new* problems.

## The Math Behind It (Class 12)
We are fitting a Polynomial function to data:
$$ y = w_0 + w_1x + w_2x^2 + ... + w_nx^n $$

*   **Underfitting ($n=1$):** A straight line ($y = mx + c$). It cannot curve to hit the points.
*   **Overfitting ($n=12$):** A crazy curve. It passes through every point perfectly, but the function is wild. The error on Training Data is 0, but on Test Data it is huge.
*   **MSE (Mean Squared Error):** How we measure error.
    $$ MSE = \frac{1}{N} \sum (y_{actual} - y_{predicted})^2 $$

## Connection to LLMs
*   **Overfitting:** If an AI memorizes the internet, it can't write a *new* story. It can only copy-paste Wikipedia.
*   **Underfitting:** A small AI (like on your phone) might be too simple to understand complex jokes.
*   **The Goal:** We want AI to understand *patterns* (Grammar, Logic), not just memorize sentences.

## What to Run
(Run these and close the plot window to see the next one)
1.  `BiasVariancePlotter.py`:
    *   **Degree 1:** A straight line. Too dumb.
    *   **Degree 12:** A crazy wiggly line. It hits every dot but looks ridiculous. This is **Overfitting**.
2.  `EpsilonPlotter.py`: Shows that some errors (noise) can never be fixed. Don't stress about perfection.
