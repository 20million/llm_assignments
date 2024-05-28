def longestCommonSubsequence(X, Y):
    m = len(X)
    n = len(Y)

    # Create a 2D array to store the length of LCS
    # Initialize the matrix with zeros
    lcsTable = [[0] * (n + 1) for i in range(m + 1)]

    # Build the lcsTable[m+1][n+1] in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                lcsTable[i][j] = 0
            elif X[i-1] == Y[j-1]:
                lcsTable[i][j] = lcsTable[i-1][j-1] + 1
            else:
                lcsTable[i][j] = max(lcsTable[i-1][j], lcsTable[i][j-1])

    # lcsTable[m][n] contains the length of LCS for X[0..m-1], Y[0..n-1]
    # Now, we will reconstruct the LCS from the lcsTable
    index = lcsTable[m][n]
    lcs = [""] * (index + 1)
    lcs[index] = ""

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:

        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i-1] == Y[j-1]:
            lcs[index-1] = X[i-1]
            i -= 1
            j -= 1
            index -= 1

        # If not same, then find the larger of two and
        # go in the direction of the larger value
        elif lcsTable[i-1][j] > lcsTable[i][j-1]:
            i -= 1
        else:
            j -= 1

    return "".join(lcs)

# usage:
X = "Ram weds Sita"
Y = "Ravan Loved Sita"
print("Longest Common Subsequence of", X, "and", Y, "is", longestCommonSubsequence(X, Y))
