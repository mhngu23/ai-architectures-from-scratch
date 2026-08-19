"""Sigmoid activation: f(x) = 1 / (1 + exp(-x)). The final layer of
`TabularTransformer`/`TextClassifierTransformer` (turns logits into
probabilities) before computing `BCELoss` — used for binary
classification.
"""
import numpy as np

class Sigmoid:
    def __call__(self, x):
        self.last_input = x
        self.last_output = 1 / (1 + np.exp(-x))
        return self.last_output

    def backward(self, grad_output):
        return grad_output * self.last_output * (1 - self.last_output)
