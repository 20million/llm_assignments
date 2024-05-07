import random
import matplotlib.pyplot as plt

# Given list `p`
p = [1, 2, 4, 6, 11, 3]

# Compute PMF and CMF
pmf = []
cmf = []
cmf_sum = 0

# Compute PMF
for value in p:
    pmf_value = value / sum(p)
    pmf.append(pmf_value)

# Compute CMF
for pmf_value in pmf:
    cmf_sum += pmf_value
    cmf.append(cmf_sum)

# Print sample and assert sum of PMF
print(f"Sample = {p}")
print(f"Sum of PMF = {sum(pmf)}")

print(f"PMF = {pmf}")
print(f"CMF = {cmf}")

# Number of random values to generate
num_random_values = 6

# Generate random values and find closest CMF values
random_values = []
closest_cmf_values = []

for _ in range(num_random_values):
    # Generate a random value from a uniform distribution
    random_value = random.uniform(0, 1)
    random_values.append(random_value)
    
    # Initialize the closest CMF value
    closest_cmf_value = None
    
    # Find the closest CMF value greater than the random value
    for cmf_value in cmf:
        if cmf_value > random_value:
            closest_cmf_value = cmf_value
            break
            
    closest_cmf_values.append(closest_cmf_value)
    print(f"Random value: {random_value}")
    print(f"Closest CMF value greater than {random_value}: {closest_cmf_value}")

# Plotting the CMF
plt.plot(cmf, label='CMF', color='blue')

# Plot random values on x-axis and closest CMF values on y-axis
plt.scatter(random_values, closest_cmf_values, color='green', marker='o', label='Closest CMF')

# Add labels and title to the plot
plt.xlabel('Random Values')
plt.ylabel('CMF and Closest CMF')
plt.title('Random Values and Closest CMF')

# Add a legend
plt.legend()

# Display the plot
plt.show()
