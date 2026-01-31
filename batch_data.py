import numpy as np

X = [[1,2,3,2.5],
     [2,3,4,3.5],
     [3,4,5,4.5]]

wt = [[0.1,0.1,0.1,0.1],
      [0.1,0.1,0.1,0.1],
      [0.1,0.1,0.1,0.1],
      ]

bias = [1,2,3]

output = np.dot(X, np.array(wt).T) + bias
print(output)