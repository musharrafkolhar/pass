import matplotlib.pyplot as plt # just for plotting graphs
import numpy as np # for calculating exponential and logartihmic terms
import pandas as pd


# Step function 
#if input x >= 0 then output = 1 
#if input x < 0 then output = 0

def step(x):
    if x >=0 :
        return 1
    else: 
        return 0
    
x = np.arange(-6,6,0.01)
step_output = [step(i) for i in x]
# plotting 
fig , ax = plt.subplots(figsize=(9,5))
ax.plot(x,step_output, color ="blue", linewidth = 3, label = "step" )
ax.legend(loc="upper left", frameon = False)
fig.show()




# Sigmoid function 
#s(x) = 1/(1+e^-x)

def sigmoid(x):
    s = (1/(1+np.exp(-x)))
    return s

# same plotting method 
x = np.arange(-6,6,0.01)
fig , ax = plt.subplots(figsize=(9,5))
ax.plot(x,sigmoid(x), color = "red", linewidth = 3, label = "sigmoid")
ax.legend(loc="upper left",frameon = False)
fig.show()




# Tanh 
#t = e^z - e^-z/e^z + e^-z

def tan(z):
    t = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return t 

z = np.arange(-6,6,0.01)
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x,tan(x), color = "blue", linewidth = 3, label = "tanh")
ax.legend(loc="upper left",frameon = False)
fig.show()



# ReLu
## linear for positive values 
## for negative values zero output 
#y = max(0,x)
def relu(x):
    y = max(0,x)
    return y

x = np.arange(-6,6,0.01)
relu_output = [relu(i)  for i in x ]
fig , ax = plt.subplots(figsize = (9,5))
ax.plot(x,relu_output , color = "green", linewidth = 3, label = "ReLU")
ax.legend(loc="upper left", frameon = False)
fig.show()


# Leaky ReLU
#f(x) = max(0.1x,x)

def leaky_relu(x):
    y = max(0.1*x,x)
    return y
x = np.arange(-6,6,0.01)
leaky_output = [leaky_relu(i) for i in x]
fig , ax = plt.subplots(figsize=(9,5))
ax.plot(x,leaky_output , color = "blue", linewidth = 3, label="Leaky_relu")
ax.legend("upper left", frameon= False)
fig.show()





