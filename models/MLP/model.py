import numpy as np

from utils.layers.linear import Linear
from utils.activations.ReLU import ReLU
from utils.activations.Sigmoid import Sigmoid


class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
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
            # activation = 'relu' if i < len(layer_dims) - 2 else None
            # activation = activation if i < len(layer_dims) - 2 else None
            activation = activation.lower() if activation else None
            self.layers.append(Linear(layer_dims[i], layer_dims[i+1], activation=activation))
            if activation == 'relu':
                self.activations.append(ReLU())
            if activation == 'sigmoid':
                self.activations.append(Sigmoid())
            else:
                self.activations.append(None)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = layer(x)
            if activation:
                x = activation(x)
        return x

    def backward(self, grad_output):
        """Performs backpropagation through the network.
            Inverse order of the computation graph is used.
        Args:
            grad_output: The gradient of the loss with respect to the output of the network.
        Returns:
            grad_output: The gradient of the loss with respect to the input of the network.
        """
        for layer, activation in reversed(list(zip(self.layers, self.activations))):
            if activation:
                grad_output = activation.backward(grad_output)
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.W)
            params.append(layer.b)
        return params

    def __call__(self, x):
        return self.forward(x)