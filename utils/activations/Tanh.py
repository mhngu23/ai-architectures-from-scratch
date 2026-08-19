"""Tanh activation: f(x) = tanh(x), values in (-1, 1). Not used directly by
the Transformer in this repo (the Transformer uses ReLU for the FFN and
Sigmoid for the classification head) — available as a general-purpose
activation for other models (e.g. MLP) in the repo.
"""
import numpy as np

class Tanh:
    def __call__(self, x):
        self.last_input = x
        self.last_output = np.tanh(x)
        return self.last_output

    def backward(self, grad_output):
        return grad_output * (1 - self.last_output ** 2)
