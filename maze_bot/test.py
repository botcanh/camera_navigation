import numpy as np

# Initialize X_batch as an empty array with 3 columns
X_batch = np.empty((0, 3), dtype=np.float64)

# Let's say we have some data to add
data = np.array([[1, 2, 3]])

# Add the data to X_batch
X_batch = np.append(X_batch, data, axis=0)

# Now, if you print X_batch, it will contain the data
print(X_batch)
