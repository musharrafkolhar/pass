import numpy as np
x1 = np.array([1,1,-1,-1])
y1=np.array([1,-1])

x2 = np.array([-1,-1,1,1])
y2 = np.array([-1,1])


# storing in weight matrix 
w = np.outer(x1,y1) + np.outer(x2,y2)
print("Trained weight matrix ")
print(w)
print("\n")

# activation function 
def activate(vector):
    return np.where(vector>=0,1,-1)

print("Forward call")
out_y1 = activate(np.dot(x1,w))
print(f"input x1{x1} --> Gives output {out_y1} that matches y1 {y1}")

out_y2 = activate(np.dot(x2,w))
print(f"input x2{x2} --> Gives output {out_y2} that matches y1 {y2}")

print("Backward call")
out_x1 = activate(np.dot(y1,w.T))
out_x2 = activate(np.dot(y2,w.T))
print(f"input y1{y1} --> Gives output {out_x1} that matches x1 {x1}")
print(f"input y2{y2} --> Gives output {out_x2} that matches x1 {x2}")