import numpy as np
import matplotlib.pyplot as plt

# these are the inputs 
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]) 

y = np.array([0,1,1,1])

# intialize weights and bias 
weights = np.zeros(2) # --> [0.,0.]
b = 0
lr = 0.1
epochs = 10

# step function 
def step(x):
    return 1 if x>=0 else 0

# Training perceptron law 
for _ in range(epochs):
    for i in range(len(X)):
        linear = np.dot(X[i],weights) + b
        y_pred = step(linear)
        error = y[i] - y_pred

        # learning rule 
        weights += error * lr * X[i]
        b += lr * error

print("Weights: ", weights)
print("Bias: ", b)

# plotting decision boundary 
plt.scatter(X[:,0], X[:,1], c=y) # X[:,0] all rows of column zero X[:,1] all rows of column y 

# Decision boundary line : w1*x1 + w2* x2 + b = 0
# w1*x1 + w2*x2 + b=0 therefore x2 that is y_vals = -(w1*x1 + b)/w2
# x1 is x_vals , x2 is y_vals , w1 is weights[0] and w2 is weights[1] 
x_vals = np.linspace(-0.5,1.5,100)
y_vals = (-weights[0]*x_vals)/weights[1]

plt.plot(x_vals, y_vals)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron Decision Boundary (OR Gate)")
plt.grid()

plt.show()