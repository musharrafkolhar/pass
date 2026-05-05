import pandas as pd
import numpy as np
def mp_neuron(x1,x2,w1,w2,threshold):
    # summation formula in the network 
    y_in = (x1*w1) + (x2*w2)
    # activation ( applying threshold)
    if y_in >= threshold:
        return 1
    else :
        return 0

#-----execution------
w1=1
w2=-1
threshold = 1
inputs = [
    (0,0),
    (0,1),
    (1,0),
    (1,1)
]
print("x1 | x2 | Output(Y)")
print("---|----|----------|")

for x1,x2 in inputs:
    output = mp_neuron(x1,x2,w1,w2,threshold)
    print(f"{x1}.  |{x2}.  |{output}")


