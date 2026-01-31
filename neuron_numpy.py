import numpy as np

X = [1,2,3,4]
wt = [0.1,-0.5,0.4,0.2]
bias = 2

output = np.dot(X,wt) + bias

print(output)