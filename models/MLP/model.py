"""MLP (Multi-Layer Perceptron) đơn giản — model baseline KHÔNG thuộc
Transformer, dùng để so sánh accuracy với `TabularTransformer`/
`TextClassifierTransformer` trên cùng dữ liệu (xem `notebooks/diabetes_demo.ipynb`
và `notebooks/transformer_demo.ipynb`).
"""
import numpy as np

from utils.layers.linear import Linear
from utils.activations.ReLU import ReLU
from utils.activations.Sigmoid import Sigmoid


class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        self.layers = []
        self.activations = []

        layer_dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(layer_dims) - 1):
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
