from unittest import result

import numpy as np
# fake_image = np.array([0, 255, 128, 20])
# # print(fake_image)
# # ---------------------------------------------------------------------------------
# # We tell numpy to rearrange the 4 numbers into 2 rows and 2 columns
# grid_image = fake_image.reshape(2, 2)

# print(grid_image)
# print(grid_image.shape)

#  # ---------------------------------------------------------------------------------

# # A flat list of 8 pixels (enough for two 2x2 images)
# fake_batch = np.array([0, 255, 128, 20, 255, 0, 50, 100])

# # # CHALLENGE: Reshape this into a batch of 2 images, where each image is 2x2
# # # Hint: You need three numbers inside your reshape: (number_of_images, rows, columns)
# batch_image = fake_batch.reshape(2, 2, 2)

# print("Our 3D Batch Layout:")
# print(batch_image)
# print("Shape:", batch_image.shape)
 

# fake_batch = np.array([0, 255, 128, 20, 255, 0, 50, 100])
# batch_image = fake_batch.reshape(2, 2, 2)

#  # ---------------------------------------------------------------------------------

# # # Normalization step
# # # We divide by 255.0 (the decimal .0 forces Python to convert integers to floats)
# normalized_batch = batch_image / 255.0

# print("Normalized Decimal Batch:")
# print(normalized_batch)

#  # ---------------------------------------------------------------------------------
# # A single image with 3 input pixels
# X = np.array([0.5, 0.8, 0.1])
# print("Input Pixels (X):", X)
# print("Input Shape:", X.shape)

# # Random weights connecting 3 inputs to 2 neurons
# W = np.array([
#     [0.2, 0.7],  # Connections for Pixel 1
#     [0.5, 0.1],  # Connections for Pixel 2
#     [0.9, 0.3]   # Connections for Pixel 3
# ])

# print("\nWeight Matrix (W):\n", W)
# print("Weight Shape:", W.shape)

# B = np.array([0.1, 0.2])  # Bias for Neuron 1 and Neuron 2

# result = np.dot(X, W) + B  # Matrix multiplication and adding 
# print("\nResult after applying weights and bias:", result)

# # # Input Pixels (X): [0.5 0.8 0.1]
# # Input Shape: (3,)

# # Weight Matrix (W):
# #  [[0.2 0.7]
# #  [0.5 0.1]
# #  [0.9 0.3]]
# # Weight Shape: (3, 2)

# # Result after applying weights and bias: [0.69 0.66]

# # ---------------------------------------------------------------------------------
 
# Imagine these are raw scores (Z) from 4 different neurons in a hidden layer
# Z = np.array([1.45, -0.23, 0.88, -5.12])
# print("Raw Neuron Scores (Z):", Z)

# # CHALLENGE: We want to apply the ReLU rule manually using NumPy.
# # NumPy has a built-in function called np.maximum(0, array) 
# # which compares every single element in the array against 0.

# activated_output = np.maximum(0, Z)

# print("\nActivated Output after ReLU:")
# print(activated_output)

# Imagine these are the final output scores (often called 'logits') for 3 classes: [Digit 3, Digit 5, Digit 6]
# logits = np.array([2.0, 1.0, 0.1])
# print("Raw Output Logits:", logits)

# # Step 1: Exponential Amplification
# exp_scores = np.exp(logits)
# print("\nAmplified Exponential Scores:", exp_scores)

# # Step 2: Normalize by dividing by the total sum
# probabilities = exp_scores / np.sum(exp_scores)

# print("\nFinal Softmax Probabilities:")
# print(probabilities)
# print("Total Sum of Percentages:", np.sum(probabilities))


# # The probabilities output by your Softmax layer yesterday
# predicted_probs = np.array([0.49900114, 0.40243297, 0.09856589])

# # The true label is Digit 5. 
# # One-hot encoded: [Digit 3 is False, Digit 5 is True, Digit 6 is False]
# true_target = np.array([0.0, 1.0, 0.0])

# # STEP 1: Extract the probability of the correct class
# # Multiplying by the target vector zeroes out the incorrect classes
# correct_class_prob = np.sum(predicted_probs * true_target)
# print("AI's confidence on the correct class:", correct_class_prob)

# # STEP 2: Calculate Cross-Entropy Loss
# # We add a tiny number (1e-15) inside the log to prevent crashing if a probability is exactly 0
# loss = -np.log(correct_class_prob + 1e-15)

# print("\nFinal Cross-Entropy Loss Score:")
# print(loss)
#  # ---------------------------------------------------------------------------------
# Let's see a real optimization climb!
# current_weight = 0.4  # Start with a poor, random weight
# target_output = 1.0   # We want our network to eventually output a perfect 1.0
# learning_rate = 0.1

# print("--- STARTING ACTUAL WEIGHT OPTIMIZATION ---")

# for epoch in range(1, 11):
#     # Simple forward pass simulation: Input (2.0) * Weight
#     simulated_prediction = 2.0 * current_weight
    
#     # Calculate how far off we are (Error)
#     error = simulated_prediction - target_output
#     loss = error ** 2  # Squared error loss
    
#     # Calculus tells us the gradient for this weight is: 2 * error * input
#     gradient = 2 * error * 2.0
    
#     # The Gradient Descent Rule: Adjust the knob!
#     current_weight = current_weight - (learning_rate * gradient)
    
#     # Print progress every 10 steps so we don't spam the screen
#     print(f"Epoch {epoch:3d} | Prediction: {simulated_prediction:.4f} | Weight Knob: {current_weight:.4f} | Loss: {loss:.4f}")
 #  # ---------------------------------------------------------------------------------
# 1. Input Data (3 pixels)
# X = np.array([0.5, 0.8, 0.1])

# # 2. LAYER 1: 3 Inputs -> 4 Neurons
# W1 = np.array([
#     [0.2, 0.4, 0.6, 0.8],
#     [0.1, 0.3, 0.5, 0.7],
#     [0.9, 0.1, 0.2, 0.3]
# ])
# B1 = np.array([0.1, 0.1, 0.1, 0.1])

# # 3. LAYER 2: 4 Hidden Inputs -> 2 Output Neurons
# W2 = np.array([
#     [0.5, 0.2],
#     [0.1, 0.8],
#     [0.4, 0.6],
#     [0.7, 0.3]
# ])
# B2 = np.array([0.2, 0.2])

# print("--- RUNNING MULTI-LAYER DEEP FORWARD PASS ---")

# # Step A: Pass through Layer 1
# Z1 = np.dot(X, W1) + B1
# A1 = np.maximum(0, Z1)  # ReLU activation creates the features for the next layer
# print("Layer 1 Hidden Features (A1):", A1)

# Z2 = np.dot(A1, W2) + B2
# A2 = np.maximum(0, Z2)  # ReLU activation creates the features for the next layer
# print("Layer 2 Hidden Features (A2):", A2)

# # Step B: Pass Layer 1's output directly into Layer 2!
# # CHALLENGE: Write the code line below to calculate Z2 using A1, W2, and B2.
# # Then pass Z2 through a final Softmax step to get the probabilities.

# exp_scores2 = np.exp(Z2)
# probabilities2 = exp_scores2 / np.sum(exp_scores2)

# print("\nFinal Deep Network Probabilities:")
# print(probabilities2)
#  # ---------------------------------------------------------------------------------
 
class DenseLayer:
    def __init__(self, input_size, neuron_count):
        # We use fixed weights here to match our visual dry run exactly
        if input_size == 3 and neuron_count == 4:
            self.weights = np.array([
                [0.2, 0.4, 0.6, 0.8],
                [0.1, 0.3, 0.5, 0.7],
                [0.9, 0.1, 0.2, 0.3]
            ])
            self.biases = np.array([[0.1, 0.1, 0.1, 0.1]])
        else:
            self.weights = np.array([
                [0.5, 0.2],
                [0.1, 0.8],
                [0.4, 0.6],
                [0.7, 0.3]
            ])
            self.biases = np.array([[0.2, 0.2]])
        
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

# Create the visual input
X = np.array([[0.5, 0.8, 0.1]])

# Instantiate our modular blueprint layers
layer1 = DenseLayer(input_size=3, neuron_count=4)
layer2 = DenseLayer(input_size=4, neuron_count=2)

# PASS THE TORCH DOWN THE ASSEMBLY LINE
output_layer1 = layer1.forward(X)
activated_layer1 = np.maximum(0, output_layer1) # ReLU

final_scores = layer2.forward(activated_layer1)

# Softmax probability engine
exp_scores = np.exp(final_scores)
probabilities = exp_scores / np.sum(exp_scores)

print("--- REUSABLE OOP CLASS PIPELINE RESULTS ---")
print("Final Output Shape:", probabilities.shape)
print("Final Probabilities:\n", probabilities)

#  # ---------------------------------------------------------------------------------