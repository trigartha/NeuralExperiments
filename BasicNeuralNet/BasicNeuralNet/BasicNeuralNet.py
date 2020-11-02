import numpy as np   

# Takes in weighted sum of the inputs and normalizes
# them through between 0 and 1 through a sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# to calculate the adjustments
def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1))-1

print('Random starting synaptic weights: ')
print(synaptic_weights)

for iteration in range(20000):

    input_layer = training_inputs
    # With the help of Sigmoid activation function, we are able to reduce the loss during the 
    # time of training because it eliminates the gradient problem in machine learning model 
    # while training.
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # Calculate error
    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)
    # np.dot * (multiplier, multiplier) 
    synaptic_weights += np.dot(input_layer.T, adjustments)



print('Synaptic weights after traning')
print(synaptic_weights)

print('Outputs after training: ')
print(outputs)