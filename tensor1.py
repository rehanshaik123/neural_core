from unittest import result

import numpy as np
# fake_image = np.array([0, 255, 128, 20])
# # print(fake_image)

# # We tell numpy to rearrange the 4 numbers into 2 rows and 2 columns
# grid_image = fake_image.reshape(2, 2)

# print(grid_image)
# print(grid_image.shape)

 

# # A flat list of 8 pixels (enough for two 2x2 images)
# fake_batch = np.array([0, 255, 128, 20, 255, 0, 50, 100])

# # CHALLENGE: Reshape this into a batch of 2 images, where each image is 2x2
# # Hint: You need three numbers inside your reshape: (number_of_images, rows, columns)
# batch_image = fake_batch.reshape(2, 2, 2)

# print("Our 3D Batch Layout:")
# print(batch_image)
# print("Shape:", batch_image.shape)
 

# fake_batch = np.array([0, 255, 128, 20, 255, 0, 50, 100])
# batch_image = fake_batch.reshape(2, 2, 2)

# # Normalization step
# # We divide by 255.0 (the decimal .0 forces Python to convert integers to floats)
# normalized_batch = batch_image / 255.0

# print("Normalized Decimal Batch:")
# print(normalized_batch)

 
# A single image with 3 input pixels
X = np.array([0.5, 0.8, 0.1])
print("Input Pixels (X):", X)
print("Input Shape:", X.shape)

# Random weights connecting 3 inputs to 2 neurons
W = np.array([
    [0.2, 0.7],  # Connections for Pixel 1
    [0.5, 0.1],  # Connections for Pixel 2
    [0.9, 0.3]   # Connections for Pixel 3
])

print("\nWeight Matrix (W):\n", W)
print("Weight Shape:", W.shape)

B = np.array([0.1, 0.2])  # Bias for Neuron 1 and Neuron 2

result = np.dot(X, W) + B  # Matrix multiplication and adding 
print("\nResult after applying weights and bias:", result)