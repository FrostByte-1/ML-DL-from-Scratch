x = [1, 2, 3, 2.5]

wt = [[0.2, 0.8, -0.5, 1],
      [0.5, -0.91, 0.26, -0.5],
      [-0.26,-0.27,0.17,0.87]]

wt1 = wt[0]
wt2 = wt[1]
wt3 = wt[2]

bias = [2, 3, 0.5]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [
    
    # Neuron 1
    wt1[0]*x[0] + wt1[1]*x[1] + wt1[2]*x[2] + wt1[3]*x[3] + bias1,
    
    # Neuron 2
    wt2[0]*x[0] + wt2[1]*x[1] + wt2[2]*x[2] + wt2[3]*x[3] + bias2,
    
    # Neuron 3
    wt3[0]*x[0] + wt3[1]*x[1] + wt3[2]*x[2] + wt3[3]*x[3] + bias3
    
]

print(output)

## Using loops

out = []

for (weights,b) in zip(wt,bias) :
    neuron_output = 0
    for (x_i,w_i) in zip(x,weights):
        neuron_output += x_i * w_i
    neuron_output += b
    out.append(neuron_output)
    
print(out)