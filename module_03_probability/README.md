# Module 03: Probability (Rolling Dice)

## The Problem
"Why does AI give a different answer every time I ask it?"

## Simple Explanation
Computers are usually exact (2 + 2 = 4). But Human language is creative.
If I say "The sky is...", the answer could be "Blue" (90% chance), "Grey" (5% chance), or "Beautiful" (5% chance).
To speak like a human, the computer calculates these % chances and then **rolls a dice** to pick one.

## The Math Behind It (Class 12)
We assume data follows a Distribution, like the Normal (Gaussian) Distribution.
The formula for the "Bell Curve" (PDF):
$$ f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{ -\frac{1}{2}(\frac{x-\mu}{\sigma})^2 } $$

*   **$\mu$ (Mu):** The Mean (Average). The center of the curve.
*   **$\sigma$ (Sigma):** The Standard Deviation. How wide the curve is.

When we "sample", we pick a random $x$, but we pick numbers near $\mu$ much more often.

## Connection to LLMs
*   **Next Token Prediction:** ChatGPT is just a giant probability machine. It predicts the % chance of every word in the dictionary being next.
*   **Temperature:**
    *   **Temp 0:** Always pick the highest %. (Robotic, boring).
    *   **Temp 1:** Roll the dice freely. (Creative, sometimes crazy).

## What to Run
1.  `NormalDistribution.py`: The "Bell Curve". Most things in nature (like student heights) follow this shape.
2.  `RandomSampler.py`: **The most important script.**
    *   It picks random numbers based on the curve.
    *   Watch the green bars slowly build up to match the red line. This is the computer "rolling the dice" thousands of times.
