# Module 05: Sequences (Spelling & Order)

## The Problem
"How does Google know I meant 'Python' when I typed 'Pyton'?"

## Simple Explanation
Language depends on **Sequence** (Order). "Dog bites Man" is very different from "Man bites Dog", even though the words are the same.
To compare two sentences, we can't just check if they have the same words. We have to check if they are in the *same order*.
This code builds a "Grid" to find the longest chain of matching letters.

## The Math Behind It (Class 12)
This uses **Dynamic Programming** (Recurrence Relations).
We build a table $L[i][j]$ representing the match length for the first $i$ letters of word A and $j$ letters of word B.

$$
L[i][j] =
\begin{cases}
L[i-1][j-1] + 1 & \text{if letters match} \\
\max(L[i-1][j], L[i][j-1]) & \text{if letters don't match}
\end{cases}
$$

We are basically saying: "The best match so far is whatever the best match was *before this letter*... plus 1 if we match!"

## Connection to LLMs
*   **Context:** AI needs to remember what you said 5 minutes ago. It treats your conversation as one long sequence.
*   **Attention:** The grid you see in the plot is similar to how "Attention" works. The AI looks at every word and compares it to every other word to find connections.

## What to Run
1.  `StringDiffWithLCS.py`:
    *   We compare "ALGORITHM" and "ALTRUISTIC".
    *   Look at the plot. The colored squares result in "ALRIT". These are the letters they share in order.
    *   This is how "Spell Check" and "Diff" (finding changes in code) work.
