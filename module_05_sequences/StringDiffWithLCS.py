import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_00_setup.Utils import set_seed, setup_plot

"""
Concepts:
- Sequences (Order matters)
- Recurrence Relations (Dynamic Programming)
- Context

This file handles "Sequences".
- Unlike the "Bag of Words" (Features) in simple classifiers, here the ORDER determines the meaning.
- The "Grid" we build compares every part of Sequence A with Sequence B.
- This "All-vs-All" comparison is the conceptual ancestor of the "Attention Mechanism" in Transformers.
"""

def longest_common_subsequence(text1, text2):
    """
    Computes LCS using Dynamic Programming.
    Returns the DP table and the LCS string.
    """
    n, m = len(text1), len(text2)
    # dp[i][j] stores length of LCS of text1[:i] and text2[:j]
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    # Recover LCS
    lcs_str = []
    i, j = n, m
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs_str.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return dp, "".join(reversed(lcs_str))

def visualize_dp_table(dp, text1, text2):
    """
    Visualizes the DP table as a heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    # We flip the DP table upside down for more intuitive matrix view if needed,
    # but let's keep it standard (0,0 at top left).
    
    cax = ax.matshow(dp, cmap='Blues')
    fig.colorbar(cax)
    
    # Set labels
    # x-axis: text2 (columns), y-axis: text1 (rows)
    # Indices 0 to m correspond to "", char1, char2...
    
    x_labels = ['""'] + list(text2)
    y_labels = ['""'] + list(text1)
    
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    
    # Annotate values
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, str(dp[i, j]), ha='center', va='center', color='gray')

    plt.title(f"LCS DP Table\n'{text1}' vs '{text2}'")
    plt.xlabel(f"Text 2: {text2}")
    plt.ylabel(f"Text 1: {text1}")
    plt.show()

if __name__ == "__main__":
    print("Running LCS (String Diff) visualization...")
    
    str1 = "ALGORITHM"
    str2 = "ALTRUISTIC"
    
    dp_table, lcs = longest_common_subsequence(str1, str2)
    
    print(f"String 1: {str1}")
    print(f"String 2: {str2}")
    print(f"LCS Length: {dp_table[-1, -1]}")
    print(f"LCS: {lcs}")
    
    visualize_dp_table(dp_table, str1, str2)
