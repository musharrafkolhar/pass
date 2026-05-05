import numpy as np

# Step 1: Define ASCII-like binary input for digits (0–9)
# (Here using simple 4-bit binary representation for demonstration)
X = np.array([
    [0,0,0,0],  # 0
    [0,0,0,1],  # 1
    [0,0,1,0],  # 2
    [0,0,1,1],  # 3
    [0,1,0,0],  # 4
    [0,1,0,1],  # 5
    [0,1,1,0],  # 6
    [0,1,1,1],  # 7
    [1,0,0,0],  # 8
    [1,0,0,1]   # 9
])

# Step 2: Target output (Even = 0, Odd = 1)
y = np.array([0,1,0,1,0,1,0,1,0,1])

# Step 3: Initialize weights and bias
weights = np.zeros(X.shape[1])
bias = 0
learning_rate = 0.1
epochs = 20

# Step 4: Activation function
def step_function(x):
    return 1 if x >= 0 else 0

# Step 5: Training the perceptron
for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = step_function(linear_output)
        
        # Update rule
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Step 6: Testing
print("Training complete!")
print("Weights:", weights)
print("Bias:", bias)

# Step 7: Prediction
def predict(x):
    return step_function(np.dot(x, weights) + bias)

# Test all inputs
print("\nDigit Classification:")
for i in range(len(X)):
    result = predict(X[i])
    label = "Odd" if result == 1 else "Even"
    print(f"Digit {i} -> {label}")