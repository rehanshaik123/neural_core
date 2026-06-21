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
 

fake_batch = np.array([0, 255, 128, 20, 255, 0, 50, 100])
batch_image = fake_batch.reshape(2, 2, 2)

# Normalization step
# We divide by 255.0 (the decimal .0 forces Python to convert integers to floats)
normalized_batch = batch_image / 255.0

print("Normalized Decimal Batch:")
print(normalized_batch)