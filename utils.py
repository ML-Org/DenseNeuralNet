import numpy as np
class L2():
    def __int__(self, regularization_constant=0.1):
        self._lambda = regularization_constant
def mini_batches(data,batch_size=1):
    n_chunks = np.ceil(len(data)/batch_size)
    return np.array_split(data, n_chunks)

#sigmoid = lambda z: 1 / (1+ np.exp(-z))
def sigmoid(z):
    return 1/(1+ np.exp(-z))
sigmoid_derivative = lambda x: x * (1-x)

def relu(z):
    return np.maximum(0,z)

def relu_derivative(x):
    return np.where(x>=0, 1, 0)

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def tanh_derivative(a):
    return (1 - (a ** 2))

def activation_fn_derivative(activation_fn):
    if activation_fn.__name__ == "sigmoid":
        return sigmoid_derivative
    elif activation_fn.__name__ == "relu":
        return relu_derivative
    elif activation_fn.__name__ == "tanh":
        return tanh_derivative

