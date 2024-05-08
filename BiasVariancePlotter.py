import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate(func, start, stop, numberOfDataPoints):
    x = np.linspace(start, stop, numberOfDataPoints)
    return func(x)

def yGenerator(x):
    fx = {}
    for i in x:
        y = (2 * (i ** 4)) - (3 * (i ** 3)) + (7 * (i ** 2)) - (23 * i) + 8 + np.random.normal(0, 3)
        fx[i] = y
    return fx

def power(x, power):
    result = [element ** power for element in x]
    return result

def deriveBeta(x, y):
    betaValue = np.linalg.inv(x.T @ x) @ x.T @ y
    return betaValue

dataset = generate(yGenerator, -5, 5, 101)

# Convert dataset to DataFrame
df = pd.DataFrame(dataset.items(), columns=['x', 'y'])

# Split the dataset into training and testing datasets (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

# Extract x and y values from the training dataset
train_x_values = train_df['x'].to_numpy()
train_y_values = train_df['y'].to_numpy()

# Calculate powers of x_values in the training data
train_x0_values = power(train_x_values, 0)
train_x1_values = train_x_values  # Equivalent to power(train_x_values, 1)
train_x2_values = power(train_x_values, 2)
train_x3_values = power(train_x_values, 3)
train_x4_values = power(train_x_values, 4)

train_x0_matrix = np.array(train_x0_values)
train_x1_matrix = np.array(train_x1_values)
train_x2_matrix = np.array(train_x2_values)
train_x3_matrix = np.array(train_x3_values)
train_x4_matrix = np.array(train_x4_values)

# Convert y_values to a numpy array
train_y_matrix = np.array(train_y_values)

print(f"x0_matrix :\n{train_x0_matrix}")
print(f"x1_matrix :\n{train_x1_matrix}")
print(f"x2_matrix :\n{train_x2_matrix}")
print(f"x3_matrix :\n{train_x3_matrix}")
print(f"x4_matrix :\n{train_x4_matrix}")
print(f"y_matrix:\n{train_y_matrix}")

print(f"beta_x0:\n{deriveBeta(train_x0_matrix, train_y_matrix)}")

