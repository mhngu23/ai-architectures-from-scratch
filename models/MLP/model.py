import numpy as np

from utils.layers.linear import Linear
from utils.activations.ReLU import ReLU

class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        input_dim: number of input features
        hidden_dims: list of hidden layer sizes, e.g., [64, 32]
        output_dim: number of output neurons
        """
        self.layers = []
        self.activations = []

        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
            # Activation after each layer except the last
            activation = 'relu' if i < len(layer_dims) - 2 else None
            self.layers.append(Linear(layer_dims[i], layer_dims[i+1], activation=activation))
            if activation == 'relu':
                self.activations.append(ReLU())
            else:
                self.activations.append(None)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation:
                x = activation(x)
        return x

    def backward(self, grad_output, lr=1e-3):
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            if activation:
                grad_output = activation.backward(grad_output)
            grad_output = layer.backward(grad_output, lr)
        return grad_output

    def __call__(self, x):
        return self.forward(x)