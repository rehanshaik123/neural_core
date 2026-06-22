# import numpy as np

# # The reusable blueprint for ANY dense neural layer
# class DenseLayer:
#     def __init__(self, input_size, neuron_count):
#         # Initialize random weights and zeroed biases automatically based on sizes
#         self.weights = np.random.randn(input_size, neuron_count) * 0.1
#         self.biases = np.zeros((1, neuron_count))
        
#     def forward(self, inputs):
#         # Calculate the forward signal flow
#         return np.dot(inputs, self.weights) + self.biases



 

# # CHALLENGE: Use our new class blueprint to instantiate two stacked layers!
# # Layer 1 takes 3 inputs and outputs to 4 hidden neurons
# layer1 = DenseLayer(input_size=3, neuron_count=4)

# # Layer 2 takes 4 hidden inputs and outputs to 2 final output neurons
# layer2 = DenseLayer(input_size=4, neuron_count=2)

# input_image=np.array([[0.8, 0.05, 0.15]])
# # Step A: Pass through Layer 1
# Z1 = layer1.forward(input_image)
# A1 = np.maximum(0, Z1)  # ReLU activation creates the features
# # Step B: Pass Layer 1's output directly into Layer 2!
# Z2 = layer2.forward(A1)
# A2 = np.maximum(0, Z2)  # ReLU activation creates the features for the next layer
# # Final Softmax step to get the probabilities.
# exp_scores2 = np.exp(Z2)
# probabilities = exp_scores2 / np.sum(exp_scores2)
# print("\nFinal Deep Network Probabilities:")
# print(probabilities)

# import numpy as np

# # Reusable Dense Layer blueprint
# class DenseLayer:
#     def __init__(self, input_size, neuron_count):
#         self.weights = np.random.randn(input_size, neuron_count) * 0.1
#         self.biases = np.zeros((1, neuron_count))
        
#     def forward(self, inputs):
#         return np.dot(inputs, self.weights) + self.biases

# # The Master Brain Container
# class NeuralNetwork:
#     def __init__(self):
#         self.layer1 = DenseLayer(input_size=3, neuron_count=4)
#         self.layer2 = DenseLayer(input_size=4, neuron_count=2)
        
#     def forward_pass(self, X):
#         z1 = self.layer1.forward(X)
#         a1 = np.maximum(0, z1) # ReLU
#         z2 = self.layer2.forward(a1)
        
#         # Softmax probability engine
#         exp_scores = np.exp(z2)
#         probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
#         return probabilities

#     def calculate_loss(self, predictions, targets):
#         # Categorical Cross-Entropy Loss
#         # We clip predictions slightly to avoid log(0) errors
#         predictions = np.clip(predictions, 1e-15, 1.0 - 1e-15)
#         loss = -np.sum(targets * np.log(predictions)) / targets.shape[0]
#         return loss

#     def update_weights_simulation(self, learning_rate=0.01):
#         # Simulated Gradient Descent Step
#         # In a full backpropagation engine, we calculate exact gradients here.
#         # For now, let's simulate turning the weight knobs slightly down the slope!
#         self.layer1.weights -= learning_rate * np.random.randn(*self.layer1.weights.shape) * 0.1
#         self.layer2.weights -= learning_rate * np.random.randn(*self.layer2.weights.shape) * 0.1
#         print("-> Weights successfully tweaked via Simulated Gradient Descent!")

# print("--- EXECUTING COMPLETE CONTAINER MODEL ---")

# # 1. Setup Input and True Targets
# # Image 1 target is Class 1 [1, 0]. Image 2 target is Class 2 [0, 1].
# X_batch = np.array([
#     [0.5, 0.8, 0.1],
#     [0.2, 0.4, 0.6]
# ])
# Y_true = np.array([
#     [1.0, 0.0],
#     [0.0, 1.0]
# ])

# # 2. Instantiate our network
# model = NeuralNetwork()

# # 3. RUN THE PIPELINE ENGINE
# probs = model.forward_pass(X_batch)
# initial_loss = model.calculate_loss(probs, Y_true)

# print("Initial Loss Score:", initial_loss)

# # 4. Turn the knobs!
# model.update_weights_simulation(learning_rate=0.1)

import numpy as np

# (Keep your DenseLayer and NeuralNetwork classes exactly the same as before)
class DenseLayer:
    def __init__(self, input_size, neuron_count):
        # We will use a fixed seed so your random numbers don't change every run
        np.random.seed(42)
        self.weights = np.random.randn(input_size, neuron_count) * 0.5
        self.biases = np.zeros((1, neuron_count))
        
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class NeuralNetwork:
    def __init__(self):
        self.layer1 = DenseLayer(input_size=3, neuron_count=4)
        self.layer2 = DenseLayer(input_size=4, neuron_count=2)
        
    def forward_pass(self, X):
        self.z1 = self.layer1.forward(X)
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.layer2.forward(self.a1)
        
        exp_scores = np.exp(self.z2)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probabilities

    def calculate_loss(self, predictions, targets):
        predictions = np.clip(predictions, 1e-15, 1.0 - 1e-15)
        return -np.sum(targets * np.log(predictions)) / targets.shape[0]

    def real_backprop_step(self, X, y, probs, lr=0.1):
        # This is the actual calculus engine that figures out exactly how to shift the weights
        m = y.shape[0]
        
        # Calculate how wrong the output layer was
        dZ2 = probs - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        dB2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Pass the blame backward to Layer 1
        dA1 = np.dot(dZ2, self.layer2.weights.T)
        dZ1 = dA1 * (self.z1 > 0) # Derivative of ReLU
        dW1 = np.dot(X.T, dZ1) / m
        dB1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Tweak the knobs using the gradient math!
        self.layer1.weights -= lr * dW1
        self.layer1.biases -= lr * dB1
        self.layer2.weights -= lr * dW2
        self.layer2.biases -= lr * dB2

# --- RUNNING THE REAL TRAINING LOOP ---
X_batch = np.array([[0.5, 0.8, 0.1], [0.2, 0.4, 0.6]])
Y_true = np.array([[1.0, 0.0], [0.0, 1.0]])

model = NeuralNetwork()

# 1. Print Initial State
initial_probs = model.forward_pass(X_batch)
print(f"Initial Loss Score: {model.calculate_loss(initial_probs, Y_true):.6f}")

print("\n--- STARTING TRAINING LOOP ---")
# Loop 200 times to let the network learn
for epoch in range(1, 201):
    probs = model.forward_pass(X_batch)
    loss = model.calculate_loss(probs, Y_true)
    
    # Run real backpropagation to fix the weights
    model.real_backprop_step(X_batch, Y_true, probs, lr=0.2)
    
    if epoch % 40 == 0:
        print(f"Epoch {epoch:3d} | Current Loss: {loss:.6f}")

# 2. Print Final State
final_probs = model.forward_pass(X_batch)
print(f"\nFinal Loss Score after 200 steps: {model.calculate_loss(final_probs, Y_true):.6f}")

print("\n--- THE FINAL PERFECT WEIGHTS & BIASES REPRESENTING CORRECT OUTPUT ---")
print("Layer 2 Weights Matrix:\n", model.layer2.weights)
print("Layer 2 Biases:\n", model.layer2.biases)