import numpy as np

X = [1,2,3,2.5]

wt = [[1,1,1,1],
     [1,1,1,1],
     [1,1,1,1]]

bias = [2,2,2]

output = np.dot(wt,X) + bias
print(output)