# Module 00: Setup

## The Problem
"If I play a game, the rules shouldn't change halfway."

## Simple Explanation
Imagine you are doing a Chemistry experiment. If you use 5g of salt today, and 10g tomorrow, you will get different results. You'll never know if you did it right.
Computer "Randomness" is actually calculated Math. If we tell the computer "Start calculating from number 42", it will produce the **same** random numbers every time. This is called **Seeding**.

## Connection to LLMs (Like ChatGPT)
*   **Training:** It costs millions of dollars to train AI. If the computer crashes, we need to restart from the *exact* same spot. We can't have "random" changes.
*   **Creativity:** When you ask ChatGPT to "be creative", it just changes its random seed/temperature.

## What to Run
1.  **Install Rules:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Test Run:**
    ```bash
    python Utils.py
    ```
    *   If you see a graph and "Random seed set to 42", you are ready!
